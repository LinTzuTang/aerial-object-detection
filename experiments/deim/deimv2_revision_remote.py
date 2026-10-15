"""Revised DEIM recipe, followed by matched original/CLAHE continuations."""
import copy
import gc
import json
import math
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import deimv2_l_remote as base
from postprocess_policy import dump, file_sha

ROOT = base.ROOT
OUT = base.OUT
COCO_ROWS = {0: 0, 1: 0, 2: 1, 3: 2, 5: 7, 8: 5, 9: 3}


def initialize(cfg, plan, device='cuda', pretrained_path=None):
    from safetensors.torch import load_file
    from engine.data.dataset.coco_dataset import mscoco_category2label, mscoco_category2name
    names = {mscoco_category2label[k]: v for k, v in mscoco_category2name.items()}
    assert {i: names[i] for i in (0, 1, 2, 3, 5, 7)} == {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    model = cfg.model
    pretrained_path = Path(pretrained_path) if pretrained_path else ROOT / 'pretrained/model.safetensors'
    state = load_file(str(pretrained_path), device='cpu')
    assert file_sha(pretrained_path) == plan['pretrained_sha256']
    current = model.state_dict(); loaded = {}; transferred = []; rebuilt = []; aliases = {}
    for name, value in current.items():
        if name in state and value.shape == state[name].shape:
            loaded[name] = state[name]
        elif name in ('decoder.up', 'decoder.reg_scale') and name not in state:
            shared = [k for k, v in current.items() if v.data_ptr() == value.data_ptr() and k in state]
            assert shared and state[shared[0]].shape == value.shape
            loaded[name] = state[shared[0]]; aliases[name] = shared[0]
        elif name in ('decoder.anchors', 'decoder.valid_mask'):
            rebuilt.append(name)
        else:
            assert name.startswith(('decoder.enc_score_head.', 'decoder.dec_score_head.', 'decoder.denoising_class_embed.')), name
            assert name in state and state[name].shape[1:] == value.shape[1:]
            mapped = value.clone()
            for target, source in COCO_ROWS.items():
                mapped[target].copy_(state[name][source])
            if name == 'decoder.denoising_class_embed.weight':
                assert value.shape[0] == 11 and state[name].shape[0] == 81
                mapped[10].copy_(state[name][80])
            else:
                assert value.shape[0] == 10 and state[name].shape[0] == 80
            loaded[name] = mapped; transferred.append(name)
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    assert sorted(missing) == sorted(rebuilt) and not unexpected
    dump(OUT / 'initialization.json', {'pretrained_sha256': plan['pretrained_sha256'],
        'semantic_COCO_rows': COCO_ROWS, 'unmapped_random_classes': [4, 6, 7],
        'person_row_copied_to_both_pedestrian_and_people': True,
        'transferred_classification_tensors': transferred, 'rebuilt_buffers': rebuilt,
        'shared_aliases': aliases, 'parameters': sum(p.numel() for p in model.parameters())})
    return model.to(device)


def optimizer_for(model, plan):
    import torch
    hp = plan['training']; groups = {}; names = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = 'backbone' if 'backbone.dinov3.' in name else (
            'classification' if name.startswith(('decoder.enc_score_head.', 'decoder.dec_score_head.', 'decoder.denoising_class_embed.')) else 'detection')
        no_decay = p.ndim <= 1 or 'norm' in name or '.bn' in name
        key = (kind, no_decay); groups.setdefault(key, []).append(p); names.setdefault(key, []).append(name)
    dump(OUT / 'optimizer_groups.json', [{'kind': k, 'no_decay': nd, 'base_lr': hp['lr_' + k],
        'parameter_names': names[k, nd]} for k, nd in groups])
    return torch.optim.AdamW([{'params': ps, 'lr': hp['lr_' + k], 'base_lr': hp['lr_' + k],
        'weight_decay': 0. if nd else hp['weight_decay']} for (k, nd), ps in groups.items()], betas=(.9, .999))


def configure(plan):
    import yaml
    path = base.configure(plan)
    cfg = yaml.safe_load(path.read_text())
    cfg['DEIMCriterion']['matcher']['matcher_change_epoch'] = 10000
    cfg['train_dataloader']['dataset']['transforms']['policy']['epoch'] = 10000
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


class ImagePolicy:
    """Wrap load_item so preprocessing precedes every spatial/photo transform."""
    def __init__(self, original, enabled):
        self.original = original; self.enabled = enabled
    def __call__(self, index):
        image, target = self.original(index)
        if self.enabled:
            from luminance_clahe import preprocess_pil
            image = preprocess_pil(image)
        return image, target


def reset_rng(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def snapshot(model, ema, optimizer, arm, epoch, counters, sequence, plan, partial_batch=None):
    import numpy as np
    import torch
    name = f'recovery_{sequence:03d}.pth'
    path = OUT / name
    torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
        'arm': arm, 'completed_epoch': epoch, 'counters': copy.deepcopy(counters),
        'partial_epoch_batch': partial_batch,
        'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
        'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
        'protocol_sha256': file_sha(ROOT / 'deim_protocol.json'), 'training_plan': plan['training']}, path)
    receipt = {'sequence': sequence, 'file': name, 'arm': arm, 'epoch': epoch,
        'partial_epoch_batch': partial_batch,
        'bytes': path.stat().st_size, 'sha256': file_sha(path), 'unix': time.time()}
    dump(OUT / 'snapshot.json', receipt)
    # Do not start evaluation or another epoch until this immutable recovery is local.
    deadline = json.loads((ROOT / 'runtime_budget.json').read_text())['total_hard_deadline_unix']
    while True:
        ack_path = OUT / 'snapshot_ack.json'
        ack = json.loads(ack_path.read_text()) if ack_path.exists() else {}
        if ack.get('sequence') == sequence and ack.get('sha256') == receipt['sha256']:
            break
        if time.time() > deadline - 120:
            raise TimeoutError('Verified recovery acknowledgement deadline')
        time.sleep(1)
    if sequence > 1:
        previous = OUT/f'recovery_{sequence-1:03d}.pth'
        if previous.exists(): previous.unlink()
    return path


def evaluate(model, postprocessor, images, guard, name, processed=False):
    # base.prediction reads PIL.Image.open; substitute only within this serial call.
    from PIL import Image
    from luminance_clahe import preprocess_pil
    original_open = Image.open
    if processed:
        def open_processed(*args, **kwargs):
            with original_open(*args, **kwargs) as image:
                return preprocess_pil(image)
        Image.open = open_processed
    try:
        return base.prediction(model, postprocessor, images, 'val', 1280, guard, name)
    finally:
        Image.open = original_open


def run():
    import numpy as np
    import torch
    from engine.core import YAMLConfig
    from engine.optim import ModelEMA
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'STARTED.json').exists()
    plan = json.loads((ROOT / 'deim_protocol.json').read_text()); hp = plan['training']
    budget = json.loads((ROOT / 'runtime_budget.json').read_text())
    dump(OUT / 'STARTED.json', {'unix': time.time(), 'protocol_sha256': file_sha(ROOT / 'deim_protocol.json')})
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0)
    torch.cuda.set_per_process_memory_fraction(10.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    reset_rng(hp['seed']); source = base.original_annotations(plan)
    from PIL import Image
    from luminance_clahe import preprocess
    assert file_sha(ROOT/'clahe_pixel_reference.npz') == plan['clahe_reference_sha256']
    reference = np.load(ROOT/'clahe_pixel_reference.npz'); probe_records = []
    for index, probe in enumerate(plan['clahe_pixel_probes']):
        with Image.open(ROOT/'dataset/images/train'/probe['file_name']) as image:
            rgb = np.asarray(image.convert('RGB'))
        assert np.array_equal(rgb, reference[f'rgb{index}']), 'JPEG decode differs from reference'
        pixels = preprocess(rgb, color_order='RGB')
        difference = np.abs(pixels.astype(np.float32)-reference[f'processed{index}'].astype(np.float32))
        record = {'file_name': probe['file_name'], 'max_difference': float(difference.max()),
            'mean_difference': float(difference.mean()), 'different_fraction': float((difference > 0).mean())}
        assert record['max_difference'] <= 2 and record['mean_difference'] <= .001 and record['different_fraction'] <= .0001
        probe_records.append(record)
    dump(OUT/'CLAHE_PLATFORM_CHECK.json', {'passed': True, 'exact_JPEG_decode': True,
        'records': probe_records, 'note': 'Declared sparse cross-platform OpenCV rounding tolerance; all A/B train/valid views use this same Colab backend'})
    reference.close(); del reference, difference, pixels, rgb
    cfg = YAMLConfig(str(configure(plan))); model = initialize(cfg, plan)
    criterion = cfg.criterion.cuda(); postprocessor = cfg.postprocessor.cuda()
    guard = base.Guard(plan['budget']['peak_gpu_limit_mib'])
    history = []; sequence = 0; branch = None; decisions = {}; terminal = {}
    try:
        # Densest image backward/AdamW/EMA allocation; discard all trial updates.
        loader = cfg.train_dataloader; dataset = loader.dataset
        counts = {r['id']: 0 for r in source['train']['images']}
        for a in source['train']['annotations']:
            if not a.get('iscrowd', 0): counts[a['image_id']] += 1
        ids = sorted(range(4500), key=lambda i: counts[source['train']['images'][i]['id']], reverse=True)[:2]
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        optimizer = optimizer_for(model, plan); trial_ema = ModelEMA(model, decay=.9998, warmups=500).to('cuda')
        model.train(); dataset.set_epoch(0)
        samples, targets = loader.collate_fn([dataset[i] for i in ids])
        samples = samples.cuda(); targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16): outputs = model(samples, targets=targets)
        with torch.autocast('cuda', enabled=False): losses = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=2250)
        loss = sum(losses.values()); assert torch.isfinite(loss); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp['clip_max_norm'], error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize(); guard.check()
        dump(OUT / 'preflight.json', {'passed': True, 'batch': 2, 'densest_boxes': max(counts.values()), 'peak_gpu_mib': guard.peak})
        model.zero_grad(set_to_none=True)
        del optimizer, trial_ema, samples, targets, outputs, losses, loss, loader
        gc.collect(); torch.cuda.empty_cache(); model.load_state_dict(initial); del initial
        cfg._train_dataloader = None; cfg.yaml_cfg['train_dataloader']['total_batch_size'] = 2
        dump(OUT / 'runtime_environment.json', {'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__,
            'opencv': __import__('cv2').__version__, 'batch': 2, 'accumulation': 8, 'optimizer_steps_per_epoch': 282,
            'train_images': 4500, 'valid_images': 700, 'valid_gradient_updates': False})
        counters = {'minibatches': 0, 'image_samples': 0, 'optimizer_steps': 0}
        for arm in ('revision', 'original', 'clahe'):
            if arm != 'revision' and not decisions.get('baseline_gate', {}).get('passed'):
                break
            optimizer = optimizer_for(model, plan); ema = ModelEMA(model, decay=.9998, warmups=500).to('cuda')
            if arm != 'revision':
                checkpoint = torch.load(branch, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model']); ema.load_state_dict(checkpoint['ema'])
                optimizer.load_state_dict(checkpoint['optimizer']); counters = copy.deepcopy(checkpoint['counters']); del checkpoint
                # A/B use the same independent loader seed rather than inherited worker RNG.
                reset_rng(hp['comparison_seed'])
            else:
                reset_rng(hp['seed'])
            cfg._train_dataloader = None; train_loader = cfg.train_dataloader
            train_loader.dataset.load_item = ImagePolicy(train_loader.dataset.load_item, arm == 'clahe')
            epochs = range(0, 3) if arm == 'revision' else range(3, 5)
            for epoch in epochs:
                limit = budget['pilot_deadline_unix'] if arm == 'revision' else budget['continuation_deadline_unix']
                required = max([r['epoch_seconds'] for r in history] or [1500]) + 240
                if time.time() + required > limit:
                    raise TimeoutError('Insufficient bounded time to finish another epoch and download')
                train_loader.set_epoch(epoch); model.train(); criterion.train(); optimizer.zero_grad(set_to_none=True)
                tick = time.monotonic(); n_batches = len(train_loader); loss_total = 0.; start_steps = counters['optimizer_steps']
                for i, (samples, targets) in enumerate(train_loader):
                    if time.time() >= limit - 180: raise TimeoutError('Training lease reserve reached')
                    samples = samples.cuda(non_blocking=True); targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
                    progress = epoch + (i + 1) / n_batches
                    warm = min(1., progress / hp['warmup_epochs'])
                    cosine = hp['final_lr_factor'] + (1 - hp['final_lr_factor']) * (1 + math.cos(math.pi * max(0., progress - hp['warmup_epochs']) / (5 - hp['warmup_epochs']))) / 2
                    for pg in optimizer.param_groups: pg['lr'] = pg['base_lr'] * warm * cosine
                    with torch.autocast('cuda', dtype=torch.bfloat16): outputs = model(samples, targets=targets)
                    with torch.autocast('cuda', enabled=False): losses = criterion(outputs, targets, epoch=epoch, step=i, global_step=epoch*n_batches+i, epoch_step=n_batches)
                    loss = sum(losses.values()); assert torch.isfinite(loss)
                    divisor = min(8, n_batches - (i // 8) * 8)
                    (loss / divisor).backward(); loss_total += float(loss.detach())
                    counters['minibatches'] += 1; counters['image_samples'] += len(samples)
                    if (i + 1) % 8 == 0 or i + 1 == n_batches:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), hp['clip_max_norm'], error_if_nonfinite=True)
                        optimizer.step(); optimizer.zero_grad(set_to_none=True); ema.update(model); counters['optimizer_steps'] += 1
                        # Keep a full model/EMA/AdamW fallback during long epochs.
                        if (counters['optimizer_steps']-start_steps) in (128, 256):
                            sequence += 1
                            snapshot(model, ema, optimizer, arm, epoch, counters, sequence, plan, partial_batch=i+1)
                    guard.check()
                    if i % 40 == 0:
                        dump(OUT / 'progress.json', {'arm': arm, 'stage': 'training', 'epoch': epoch + 1,
                            'batch': i + 1, 'batches': n_batches, 'counters': counters, 'loss': float(loss.detach()), 'peak_gpu_mib': guard.peak, 'unix': time.time()})
                assert counters['optimizer_steps'] - start_steps == 282
                train_seconds = time.monotonic() - tick; sequence += 1
                saved = snapshot(model, ema, optimizer, arm, epoch + 1, counters, sequence, plan)
                if arm == 'revision' and epoch == 2:
                    branch = OUT / 'branch_epoch3.pth'; shutil.copy2(saved, branch)
                del samples, targets, outputs, losses, loss; gc.collect(); torch.cuda.empty_cache()
                metrics = None
                if arm == 'revision' or epoch == 4:
                    metrics = evaluate(ema.module, postprocessor, source['val']['images'], guard, f'{arm}_epoch{epoch+1}_valid_original.json')
                if epoch == (2 if arm == 'revision' else 4):
                    torch.save({'model': ema.module.state_dict(), 'config': cfg.yaml_cfg, 'arm': arm,
                        'epoch': epoch + 1, 'protocol_sha256': file_sha(ROOT / 'deim_protocol.json')}, OUT / (arm + '_terminal.pth'))
                    terminal[arm] = {'original': metrics, 'counters': copy.deepcopy(counters)}
                    if arm != 'revision':
                        terminal[arm]['clahe'] = evaluate(ema.module, postprocessor, source['val']['images'], guard, f'{arm}_epoch5_valid_clahe.json', processed=True)
                record = {'arm': arm, 'epoch': epoch + 1, 'train_seconds': train_seconds, 'epoch_seconds': time.monotonic()-tick,
                    'mean_loss': loss_total/n_batches, 'metrics': metrics, 'counters': copy.deepcopy(counters), 'peak_gpu_mib': guard.peak}
                history.append(record); dump(OUT / 'history.json', history); dump(OUT / 'terminal_metrics.json', terminal)
                if arm == 'revision' and epoch == 2:
                    trend = metrics['all_valid']['AP'] - history[0]['metrics']['all_valid']['AP']
                    checks = {'AP_at_least_020': metrics['all_valid']['AP'] >= .2,
                        'AP_small_at_least_010': metrics['all_valid']['small_AP'] >= .1,
                        'improving_or_AP_at_least_030': trend >= .005 or metrics['all_valid']['AP'] >= .3,
                        'complete_AB_fits_budget': time.time()+4*max(r['epoch_seconds'] for r in history)*1.15+600 < budget['continuation_deadline_unix']}
                    decisions['baseline_gate'] = {'passed': all(checks.values()), 'checks': checks, 'AP_trend': trend}
                    dump(OUT / 'PILOT_GATE.json', {**decisions['baseline_gate'], 'protocol_sha256': file_sha(ROOT / 'deim_protocol.json')})
                # Earlier snapshots are regenerable only after current verified recovery.
                if sequence > 1:
                    old = OUT / f'recovery_{sequence-1:03d}.pth'
                    if old.exists(): old.unlink()
            del optimizer, ema, train_loader; gc.collect(); torch.cuda.empty_cache()
        dump(OUT / 'DECISION.json', decisions)
        # Include the immutable latest recovery and all terminal EMA weights/predictions.
        names = ['initialization.json', 'optimizer_groups.json', 'preflight.json', 'runtime_environment.json', 'CLAHE_PLATFORM_CHECK.json',
            'effective_config.yml', 'history.json', 'terminal_metrics.json', 'DECISION.json', 'PILOT_GATE.json', 'snapshot.json']
        names += [p.name for p in OUT.glob('*_terminal.pth')]
        names += [p.name for p in OUT.glob('*valid_*.json')]
        if sequence: names.append(f'recovery_{sequence:03d}.pth')
        if branch: names.append(branch.name)
        names = sorted(set(n for n in names if (OUT/n).exists()))
        dump(OUT / 'READY.json', {'unix': time.time(), 'completed_training_epochs': len(history),
            'preprocessing_comparison_completed': set(terminal) == {'revision', 'original', 'clahe'},
            'decisions': decisions, 'peak_gpu_mib': guard.peak, 'worker_source_sha256': file_sha(Path(__file__)),
            'protocol_sha256': file_sha(ROOT/'deim_protocol.json'), 'valid_gradient_updates': False, 'test_images_used': 0,
            'artifacts': {n: {'bytes': (OUT/n).stat().st_size, 'sha256': file_sha(OUT/n)} for n in names}})
    finally:
        guard.done.set(); guard.thread.join(timeout=2)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        dump(OUT/'FAILED.json', {'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(), 'unix': time.time()})
        raise
