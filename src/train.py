#!/usr/bin/env python3
"""Reproduce the initial/gentle branch or the independent Fresh1280 branch."""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

class RawDataset(YOLODataset):
    # The recorded gentle-control pipeline loads native resolution before transforms.
    def load_image(self,index,rect_mode=True):
        image=cv2.imread(self.im_files[index])
        if image is None:raise FileNotFoundError(self.im_files[index])
        return image,image.shape[:2],image.shape[:2]

class GentleTrainer(DetectionTrainer):
    dataset_type=RawDataset
    def build_dataset(self,img_path,mode="train",batch=None):
        if mode!="train":return super().build_dataset(img_path,mode,batch)
        return self.dataset_type(img_path=img_path,imgsz=self.args.imgsz,batch_size=batch,
            augment=True,hyp=self.args,rect=False,cache=None,single_cls=False,
            stride=max(int(unwrap_model(self.model).stride.max()),32),pad=0.,
            prefix="train: ",task="detect",classes=self.args.classes,data=self.data,fraction=1.)

class CLAHEDataset(RawDataset):
    def load_image(self,index,rect_mode=True):
        from luminance_clahe import preprocess
        image,original_shape,resized_shape=super().load_image(index,rect_mode)
        return preprocess(image,color_order='BGR'),original_shape,resized_shape

class MatchedOriginalTrainer(GentleTrainer):
    def _build_train_pipeline(self):
        if self.args.batch!=2:raise RuntimeError('Matched experiment requires physical batch2')
        return super()._build_train_pipeline()
    def _handle_nan_recovery(self,epoch):
        recovered=super()._handle_nan_recovery(epoch)
        if recovered:raise RuntimeError('Refuse repeated epochs in the matched comparison')
        return recovered
    def final_eval(self):pass

class MatchedCLAHETrainer(MatchedOriginalTrainer):
    dataset_type=CLAHEDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="HW1 dataset YAML (train/val only)")
    p.add_argument("--stage", choices=("initial", "gentle", "fresh", "mosaic_control", "mosaic_clean", "preprocess_original", "preprocess_clahe"), default="initial")
    p.add_argument("--weights", default="yolo26l.pt",
                    help="Official COCO-pretrained yolo26l.pt for initial stage; prior best.pt for gentle stage")
    p.add_argument("--project", default="runs/hw1")
    p.add_argument("--name", default="yolo26l_gentle_control")
    p.add_argument("--device", default="0")
    p.add_argument("--stop-after", type=int, default=66,
                   help="Fresh branch: stop after this epoch while retaining the original 100-epoch schedule")
    p.add_argument("--resume", type=Path,
                   help="Resume this same stage from its optimizer-bearing recovery_last.pt")
    args = p.parse_args()
    import ultralytics
    if ultralytics.__version__ != '8.4.173':
        raise RuntimeError('Install the pinned Ultralytics version')
    if not args.resume and (Path(args.project) / args.name).exists():
        raise FileExistsError('Use a new --name, or --resume the existing recovery checkpoint')
    if args.resume:
        import json, torch
        if args.stage.startswith('preprocess_'):
            raise ValueError('Restart a matched 5-epoch arm from Gentle to preserve equal update counts')
        state_file = args.resume.parent / 'retraining_state.json'
        if not state_file.is_file() or json.loads(state_file.read_text())['stage'] != args.stage:
            raise ValueError('Keep retraining_state.json with recovery_last.pt and use its original --stage')
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved.get('optimizer') is None or saved.get('epoch', -1) < 0:
            raise ValueError('Resume needs recovery_last.pt, not a stripped inference weight')
        if args.stage == 'fresh' and saved['epoch'] + 1 >= args.stop_after:
            raise ValueError('Fresh run has already reached --stop-after')
        del saved
    from runtime_guard import start_guard
    start_guard(args.device)
    model = YOLO(str(args.resume) if args.resume else args.weights)
    # Preserve resumable states before Ultralytics final_eval strips last.pt.
    # Export explicit 1-based epoch snapshots, never confuse best.pt with epoch37.
    def export_training_state(trainer):
        import json, shutil
        from ultralytics.utils.torch_utils import strip_optimizer
        epoch = int(trainer.epoch) + 1
        recovery = trainer.save_dir / 'recovery_last.pt'
        temporary = recovery.with_suffix('.pt.tmp')
        shutil.copy2(trainer.last, temporary); temporary.replace(recovery)
        fixed = 37 if args.stage == 'fresh' else (10 if args.stage in ('mosaic_control', 'mosaic_clean') else None)
        if epoch == fixed:
            target = trainer.save_dir / f'epoch_{epoch:03d}_ema.pt'
            shutil.copy2(trainer.last, target); strip_optimizer(str(target))
        (trainer.save_dir / 'retraining_state.json').write_text(json.dumps({
            'stage': args.stage, 'completed_epoch': epoch,
            'recovery': str(recovery), 'fixed_export_epoch': fixed,
            'note': 'Epochs are 1-based. best.pt is validation-selected; fixed export is terminal EMA.'
        }, indent=2) + '\n')
    model.add_callback('on_model_save', export_training_state)
    common = dict(data=args.data, project=args.project, name=args.name,
        device=args.device, batch=4, imgsz=960, optimizer="AdamW",
        lrf=0.01, weight_decay=0.0005, cos_lr=True, seed=3407,
        deterministic=True, amp=True, workers=4, cls_remap=True,
        mixup=0.0, cutmix=0.0, copy_paste=0.0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, val=True, save=True, plots=True)
    if args.resume:
        common['resume'] = str(args.resume)
    if args.stage == "initial":
        common.update(epochs=30, patience=8, lr0=0.000714, warmup_epochs=3,
                      mosaic=1.0, close_mosaic=5, scale=0.5, translate=0.1)
    elif args.stage == "gentle":
        common.update(epochs=20, patience=8, lr0=0.0001, warmup_epochs=1,
                      mosaic=0.0, close_mosaic=0, scale=0.2, translate=0.1)
    elif args.stage == "fresh":
        if not 1 <= args.stop_after <= 100:raise ValueError('Fresh stop epoch must be within 1..100')
        common.update(imgsz=1280,batch=2,nbs=64,epochs=100,patience=0,
            optimizer='AdamW',lr0=.0003,momentum=.9,warmup_epochs=5.,warmup_bias_lr=0.,
            mosaic=.3,close_mosaic=15,scale=.2,translate=.05,hsv_s=.3,hsv_v=.3,
            flipud=0.,cls_pw=0.,multi_scale=0.,cache=False,save_period=-1)
        def stop_at_recorded_epoch(trainer):
            if int(trainer.epoch)+1>=args.stop_after:trainer.stop=True
        model.add_callback('on_model_save',stop_at_recorded_epoch)
    elif args.stage.startswith('preprocess_'):
        common.update(imgsz=1280,batch=2,nbs=64,epochs=5,patience=0,optimizer='AdamW',lr0=.0001,lrf=.1,
            momentum=.9,warmup_epochs=1.,warmup_bias_lr=0.,mosaic=0.,close_mosaic=0,scale=.2,translate=.05,
            hsv_s=.3,hsv_v=.3,flipud=0.,cls_pw=0.,multi_scale=0.,cache=False,save_period=-1,plots=False,workers=2,seed=13407)
        import json,shutil
        from ultralytics.utils.torch_utils import strip_optimizer
        counters={'optimizer_step_calls':0,'successful_optimizer_steps':0}
        def track_updates(trainer):
            assert trainer.start_epoch==0 and not trainer.resume and len(trainer.optimizer.state)==0
            trainer.optimizer.register_step_post_hook(lambda *unused:counters.__setitem__('successful_optimizer_steps',counters['successful_optimizer_steps']+1))
            original=trainer.optimizer_step
            def step():
                counters['optimizer_step_calls']+=1
                return original()
            trainer.optimizer_step=step
        def save_terminal(trainer):
            if int(trainer.epoch)+1==5:
                target=trainer.save_dir/'terminal.pt';shutil.copy2(trainer.last,target);strip_optimizer(str(target))
                (trainer.save_dir/'matched_update_counts.json').write_text(json.dumps(counters,indent=2)+'\n')
        model.add_callback('on_train_start',track_updates);model.add_callback('on_model_save',save_terminal)
    else:
        common.update(imgsz=1280,batch=2,nbs=64,epochs=10,patience=0,
            optimizer='AdamW',lr0=.0001,momentum=.9,warmup_epochs=1.,warmup_bias_lr=0.,
            mosaic=.3 if args.stage=='mosaic_control' else 0.,close_mosaic=0,scale=.2,translate=.05,
            hsv_s=.3,hsv_v=.3,flipud=0.,cls_pw=0.,multi_scale=0.,cache=False,save_period=-1,plots=False)
    if args.stage == "initial":
        common.update(optimizer="auto", save_period=1)
        model.train(**common)
    elif args.stage == "gentle":
        common.update(warmup_bias_lr=0.0)
        model.train(trainer=GentleTrainer, **common)
    elif args.stage.startswith('preprocess_'):
        model.train(trainer=MatchedCLAHETrainer if args.stage=='preprocess_clahe' else MatchedOriginalTrainer,**common)
    else:model.train(**common)


if __name__ == "__main__":
    main()
