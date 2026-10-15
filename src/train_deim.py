#!/usr/bin/env python3
"""Portable DEIMv2-L retraining from generic COCO weights, with local recovery.

The default reconstructs the recorded LR transitions and update budget, including
the two partial passes. Interrupted Colab loader order is not exactly replayable.
This entry point never depends on an old Colab VM, transfer server or checkpoint.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'vendor/DEIMv2'))
sys.path.insert(0, str(CODE / 'experiment_sources'))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def segments(max_epochs):
    """Reconstruct 10 whole passes + 2048/1024 additional minibatches."""
    result = []
    for epoch in range(1, max_epochs + 1):
        result.append({
            'name': f'epoch_{epoch:03d}', 'epoch': epoch - 1, 'whole': True,
            'batches': 2250, 'schedule': 'revision' if epoch <= 2 else 'continuation',
            'rebase': epoch == 3,
            'elapsed': max(0, epoch - 3) + (1024 / 2250 if epoch >= 9 else 0),
        })
        if epoch in (2, 8):
            result.append({
                'name': f'prefix_after_epoch_{epoch:03d}', 'epoch': epoch, 'whole': False,
                'batches': 2048 if epoch == 2 else 1024, 'schedule': 'continuation',
                'rebase': epoch == 2, 'elapsed': 0 if epoch == 2 else 6,
            })
    return result


def configure(data_root, prepared, output):
    import yaml
    from engine.data.transforms._transforms import ConvertBoxes, ConvertPILImage
    # Compatibility shim used in the original TorchVision 0.26 training runtime.
    for transform in (ConvertBoxes, ConvertPILImage):
        transform.transform = transform._transform
    config = yaml.safe_load((CODE / 'records/deim_epoch10_effective_config.yml').read_text())
    config['DINOv3STAs']['weights_path'] = None
    config['output_dir'] = str(output)
    for split, key in [('train', 'train_dataloader'), ('val', 'val_dataloader')]:
        loader = config[key]
        loader.pop('batch_size', None)
        loader.update(total_batch_size=2 if split == 'train' else 1,
                      num_workers=0, drop_last=False, shuffle=split == 'train')
        loader['dataset'].update(img_folder=str(data_root / 'images' / split),
                                 ann_file=str(prepared / f'{split}_numeric.json'))
    path = output / 'effective_config.yml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def evaluate(model, postprocessor, data_root, prepared, output, name):
    import numpy as np
    import torch
    from PIL import Image
    source = json.loads((prepared / 'val_numeric.json').read_text())
    model.eval(); postprocessor.eval()
    mean = torch.tensor([.485, .456, .406], device='cuda')[:, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[:, None, None]
    rows = []
    for record in source['images']:
        width, height = record['width'], record['height']
        with Image.open(data_root / 'images/val' / record['file_name']) as image:
            if image.size != (width, height):
                raise ValueError('Image size differs from annotation')
            pixels = np.asarray(image.convert('RGB').resize((1280, 1280), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
        tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1).cuda()
        tensor = ((tensor - mean) / std)[None]
        with torch.inference_mode():
            pred = postprocessor(model(tensor), torch.tensor([[width, height]], device='cuda'))[0]
        for cls, score, box in zip(pred['labels'].cpu().tolist(), pred['scores'].cpu().tolist(), pred['boxes'].cpu().tolist()):
            if score < .001:
                continue
            x1, y1, x2, y2 = box
            x1, x2 = max(0., min(x1, width)), max(0., min(x2, width))
            y1, y2 = max(0., min(y1, height)), max(0., min(y2, height))
            bbox = [round(float(v), 3) for v in (x1, y1, x2 - x1, y2 - y1)]
            bbox[2] = round(min(bbox[2], width - bbox[0]), 3)
            bbox[3] = round(min(bbox[3], height - bbox[1]), 3)
            if min(bbox[2:]) > 0:
                rows.append({'image_id': record['id'], 'category_id': int(cls),
                             'score': round(float(score), 6), 'bbox': bbox})
    target = output / f'{name}_valid.json'
    dump(target, rows)
    metrics_path = target.with_suffix('.metrics.json')
    # DEIM installs faster-coco-eval globally in its process. Isolate canonical AP.
    subprocess.run([sys.executable, str(CODE / 'experiment_sources/deimv2_canonical_eval.py'),
                    '--annotation', str(prepared / 'val_numeric.json'), '--predictions', str(target),
                    '--output', str(metrics_path)], check=True)
    return json.loads(metrics_path.read_text()), target


def save_checkpoint(path, value):
    import torch
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('outputs/data'))
    parser.add_argument('--pretrained', type=Path, default=Path('outputs/pretrained/deim_model.safetensors'))
    parser.add_argument('--output', type=Path, default=Path('outputs/train/deim'))
    parser.add_argument('--mode', choices=['recorded', 'early-stop'], default='recorded')
    parser.add_argument('--resume', type=Path, help='Local recovery_last.pth from this entry point')
    args = parser.parse_args()
    import numpy as np
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Use one NVIDIA CUDA GPU with native BF16 support (the original run used L4)')
    if torch.__version__.split('+')[0] != '2.11.0':
        raise RuntimeError('Install the pinned Torch version')
    from runtime_guard import start_guard
    start_guard('0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data_root, prepared, output = args.data_root.resolve(), args.prepared.resolve(), args.output.resolve()
    preparation = json.loads((prepared / 'prepared.json').read_text())
    if preparation['data_root'] != str(data_root):
        raise ValueError('Prepared data points to a different --data-root')
    for split, digest in preparation['annotation_sha256'].items():
        if sha(data_root / 'annotations' / f'instances_{split}.json') != digest:
            raise ValueError('Original annotations changed after preparation')
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError(output)
    else:
        output.mkdir(parents=True, exist_ok=False)
    from engine.core import YAMLConfig
    from engine.optim import ModelEMA
    import deimv2_revision_remote as original_recipe
    original_recipe.OUT = output
    plan = json.loads((CODE / 'records/deim_protocol.json').read_text())
    plan['pretrained_sha256'] = 'c0d0f45fa62b785126db5b40f5530866fecaab2d340cea812d2ba5f0f5646163'
    hp = plan['training']
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    cfg = YAMLConfig(str(configure(data_root, prepared, output)))
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['mode'] != args.mode or checkpoint['annotation_sha256'] != preparation['annotation_sha256']:
            raise ValueError('Resume mode/data differs from saved run')
        model = cfg.model.cuda(); model.load_state_dict(checkpoint['model'], strict=True)
    else:
        model = original_recipe.initialize(cfg, plan, pretrained_path=args.pretrained)
    optimizer = original_recipe.optimizer_for(model, plan)
    ema = ModelEMA(model, decay=.9998, warmups=500).to('cuda')
    state = {'next_segment': 0, 'best_AP': -1., 'best_epoch': None, 'anchor_AP': None,
             'patience_count': 0, 'history': [],
             'counters': {'minibatches': 0, 'image_samples': 0, 'optimizer_steps': 0}}
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer']); ema.load_state_dict(checkpoint['ema'])
        state = checkpoint['state']
        torch.set_rng_state(checkpoint['rng_torch']); torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        random.setstate(checkpoint['rng_python']); np.random.set_state(checkpoint['rng_numpy'])
        del checkpoint; gc.collect()
        if state['best_epoch'] is not None and not (output / 'best_ema.pth').exists():
            raise FileNotFoundError('Keep best_ema.pth alongside recovery_last.pth when moving a run')
    else:
        # Original recipe reset the loader seed after the discarded memory probe.
        random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    criterion, postprocessor, loader = cfg.criterion.cuda(), cfg.postprocessor.cuda(), cfg.train_dataloader
    if len(loader) != 2250:
        raise ValueError('Expected 4500 train images, physical batch 2')
    schedule = segments(10 if args.mode == 'recorded' else 15)
    dump(output / 'retraining_protocol.json', {
        'mode': args.mode, 'segments': schedule, 'training': hp,
        'authoritative_optimization': 'This custom loop; stock optimizer/scheduler fields in effective_config are unused',
        'seed': 3407, 'workers': 0, 'batch': 2, 'accumulation': 8,
        'recorded_target_optimizer_steps': 3204 if args.mode == 'recorded' else None,
        'valid_gradient_updates': False, 'test_accessed': False,
        'scope': 'Recorded update budget/LR transitions, not exact interrupted-loader or original-weight replay',
    })
    dump(output / 'runtime_environment.json', {'torch': torch.__version__,
         'torchvision': __import__('torchvision').__version__, 'gpu': torch.cuda.get_device_name(0),
         'workers': 0, 'precision': 'BF16 forward, FP32 loss and validation',
         'pretrained_sha256': plan['pretrained_sha256']})
    reason = 'recorded_budget_completed' if args.mode == 'recorded' else 'maximum_epochs'
    for index in range(state['next_segment'], len(schedule)):
        segment = schedule[index]
        if segment['rebase']:
            for group in optimizer.param_groups:
                group['base_lr'] = group['lr']
        loader.set_epoch(segment['epoch'])
        model.train(); criterion.train(); optimizer.zero_grad(set_to_none=True)
        started = time.monotonic(); loss_sum = 0.
        for i, (images, targets) in enumerate(loader):
            fraction = (i + 1) / len(loader)
            if segment['schedule'] == 'revision':
                progress = segment['epoch'] + fraction
                factor = min(1., progress / .25) * (.1 + .9 * (1 + math.cos(math.pi * max(0., progress - .25) / 4.75)) / 2)
            else:
                progress = min(1., (segment['elapsed'] + fraction) / 13)
                factor = .1 + .9 * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            images = images.cuda(non_blocking=True)
            targets = [{key: value.cuda(non_blocking=True) for key, value in target.items()} for target in targets]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                predictions = model(images, targets=targets)
            with torch.autocast('cuda', enabled=False):
                losses = criterion(predictions, targets, epoch=segment['epoch'], step=i,
                                   global_step=segment['epoch'] * len(loader) + i, epoch_step=len(loader))
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; use the last completed local recovery')
            divisor = min(8, segment['batches'] - (i // 8) * 8)
            (loss / divisor).backward(); loss_sum += float(loss.detach())
            counters = state['counters']
            counters['minibatches'] += 1; counters['image_samples'] += len(images)
            if (i + 1) % 8 == 0 or i + 1 == segment['batches']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); ema.update(model)
                counters['optimizer_steps'] += 1
            if (i + 1) % 100 == 0:
                print(segment['name'], i + 1, '/', segment['batches'], 'updates', counters['optimizer_steps'], flush=True)
            if i + 1 == segment['batches']:
                break
        del images, targets, predictions, losses, loss
        gc.collect(); torch.cuda.empty_cache()
        metrics, prediction_path = evaluate(ema.module, postprocessor, data_root, prepared, output, segment['name'])
        ap = metrics['all_valid']['AP']
        epoch_value = segment['epoch'] + (1 if segment['whole'] else segment['batches'] / 2250)
        if ap > state['best_AP']:
            state.update(best_AP=ap, best_epoch=epoch_value)
            save_checkpoint(output / 'best_ema.pth', {'model': ema.module.state_dict(), 'config': cfg.yaml_cfg,
                                                    'epoch': epoch_value, 'retrained': True})
            shutil.copy2(prediction_path, output / 'best_valid.json')
            dump(output / 'best_metrics.json', metrics)
        if state['anchor_AP'] is None or ap >= state['anchor_AP'] + .003:
            if segment['whole'] or state['anchor_AP'] is None:
                state.update(anchor_AP=ap, patience_count=0)
        elif segment['whole']:
            state['patience_count'] += 1
        state['history'].append({'segment': segment, 'AP': ap, 'epoch': epoch_value,
                                 'seconds': time.monotonic() - started, 'mean_loss': loss_sum / segment['batches'],
                                 'counters': copy.deepcopy(state['counters'])})
        state['next_segment'] = index + 1
        save_checkpoint(output / 'recovery_last.pth', {
            'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
            'state': state, 'mode': args.mode, 'annotation_sha256': preparation['annotation_sha256'],
            'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
            'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
        })
        dump(output / 'history.json', state['history'])
        print(segment['name'], 'valid AP', ap, 'best AP', state['best_AP'], flush=True)
        if args.mode == 'early-stop' and segment['whole'] and epoch_value >= 6 and state['patience_count'] >= 2:
            reason = 'early_stopping'
            break
    dump(output / 'completed.json', {
        'mode': args.mode, 'stop_reason': reason, 'best_epoch': state['best_epoch'], 'best_AP': state['best_AP'],
        'counters': state['counters'], 'checkpoint': 'best_ema.pth',
        'sha256': sha(output / 'best_ema.pth'),
        'reported_original_public_AP_is_not_a_score_for_this_retraining': True,
    })


if __name__ == '__main__':
    main()
