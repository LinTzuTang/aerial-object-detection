"""Bounded single-model DEIMv2-L pilot and conditional continuation; no test access."""
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
import threading
import time
import traceback
from pathlib import Path

ROOT=Path('/content/cvpdl_postprocess');REPO=ROOT/'DEIMv2';OUT=ROOT/'deimv2_l'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(REPO))
from postprocess_policy import dump,file_sha


class Guard:
    def __init__(self,limit):
        import pynvml
        pynvml.nvmlInit();self.nv=pynvml;self.handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        self.limit=limit;self.peak=0;self.done=threading.Event();self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()
    def run(self):
        while not self.done.wait(.05):
            used=self.nv.nvmlDeviceGetMemoryInfo(self.handle).used/1024**2;self.peak=max(self.peak,used)
            if used>=self.limit:
                dump(OUT/'MEMORY_LIMIT.json',{'peak_gpu_mib':used,'limit_mib':self.limit,'unix':time.time()})
                os._exit(78)
    def check(self):
        if self.peak>=self.limit:raise RuntimeError('VRAM ceiling reached')


def configure(plan):
    import yaml
    from engine.core.yaml_utils import load_config
    # torchvision>=0.24 renamed custom Transform hooks; preserve the official transform bodies.
    from engine.data.transforms._transforms import ConvertBoxes,ConvertPILImage
    for cls in [ConvertBoxes,ConvertPILImage]:cls.transform=cls._transform
    cfg=load_config(str(REPO/'configs/deimv2/deimv2_dinov3_l_coco.yml'))
    cfg.pop('__include__',None)  # Already flattened; relative includes belong to the source directory.
    cfg.update(num_classes=10,remap_mscoco_category=False,eval_spatial_size=[1280,1280],use_ema=True)
    cfg['DINOv3STAs']['weights_path']=None
    cfg['DEIMCriterion']['matcher']['matcher_change_epoch']=plan['training']['epochs']-8
    cfg['ema'].update(decay=.9998,warmups=500,start=0)
    for split,key in [('train','train_dataloader'),('val','val_dataloader')]:
        loader=cfg[key];loader.pop('batch_size',None);loader['total_batch_size']=1
        loader.update(num_workers=2,drop_last=False,shuffle=split=='train',pin_memory=True)
        loader['dataset'].update(img_folder=str(ROOT/'dataset/images'/split),ann_file=str(OUT/(split+'_numeric.json')))
        ops=[]
        if split=='train':ops=[{'type':'RandomPhotometricDistort','p':.3},{'type':'RandomIoUCrop','p':.3},{'type':'RandomHorizontalFlip','p':.5}]
        ops += [{'type':'Resize','size':[1280,1280]},{'type':'SanitizeBoundingBoxes','min_size':1},
            {'type':'ConvertPILImage','dtype':'float32','scale':True},
            {'type':'Normalize','mean':[.485,.456,.406],'std':[.229,.224,.225]},
            {'type':'ConvertBoxes','fmt':'cxcywh','normalize':True}]
        loader['dataset']['transforms']={'type':'Compose','ops':ops,'policy':{'name':'stop_epoch','epoch':plan['training']['epochs']-8,
            'ops':['RandomPhotometricDistort','RandomIoUCrop']}} if split=='train' else {'type':'Compose','ops':ops}
        loader['collate_fn']={'type':'BatchImageCollateFunction','base_size':1280,'base_size_repeat':None,
            'mixup_prob':0.,'copyblend_prob':0.,'stop_epoch':plan['training']['epochs']+1}
    path=OUT/'effective_config.yml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));return path


def original_annotations(plan):
    source={}
    for split,n in [('train',4500),('val',700)]:
        path=ROOT/'dataset/annotations'/('instances_'+split+'.json')
        assert file_sha(path)==plan['annotation_sha256'][split]
        data=json.loads(path.read_text());assert len(data['images'])==n
        assert sorted(c['id'] for c in data['categories'])==list(range(10))
        assert len(list((ROOT/'dataset/images'/split).glob('*.jpg')))==n
        source[split]=data;mapping={r['id']:i+1 for i,r in enumerate(data['images'])}
        derived=copy.deepcopy(data)
        for r in derived['images']:r['id']=mapping[r['id']]
        for i,r in enumerate(derived['annotations']):r.update(id=i+1,image_id=mapping[r['image_id']])
        dump(OUT/(split+'_numeric.json'),derived)
        dump(OUT/(split+'_id_mapping.json'),mapping)
    assert not ({r['id'] for r in source['train']['images']}&{r['id'] for r in source['val']['images']})
    assert not (ROOT/'dataset/images/test').exists()
    return source


def initialize(cfg,plan):
    import torch
    from safetensors.torch import load_file
    model=cfg.model
    state=load_file(str(ROOT/'pretrained/model.safetensors'),device='cpu')
    current=model.state_dict();matched={};skipped=[];rebuilt=[];aliases={}
    for k,v in current.items():
        if k in state and state[k].shape==v.shape:matched[k]=state[k]
        else:
            if k in ['decoder.up','decoder.reg_scale'] and k not in state:
                shared=[name for name,tensor in current.items() if tensor.data_ptr()==v.data_ptr() and name in state]
                assert shared and state[shared[0]].shape==v.shape
                matched[k]=state[shared[0]];aliases[k]=shared[0];continue
            if k in ['decoder.anchors','decoder.valid_mask']:
                rebuilt.append(k);continue
            assert k.startswith(('decoder.enc_score_head.','decoder.dec_score_head.','decoder.denoising_class_embed.')),k
            skipped.append(k)
    model.load_state_dict(matched,strict=False)
    assert any(k.startswith('backbone.dinov3.') for k in matched)
    assert len(matched)>len(current)*.95
    dump(OUT/'initialization.json',{'checkpoint_sha256':file_sha(ROOT/'pretrained/model.safetensors'),
        'hf_revision':plan['hf_revision'],'repo_commit':plan['repo_commit'],'loaded_tensors':len(matched),'total_tensors':len(current),
        'reinitialized_classification_tensors':skipped,'resolution_buffers_rebuilt':rebuilt,'shared_tensor_aliases_restored':aliases,
        'class_mapping':'Original IDs0..9; no COCO category remapping',
        'source':'Official DEIMv2-L DINOv3 LVD1689M plus COCO; no task-specific pretrained weights',
        'trainable_parameters':sum(v.numel() for v in model.parameters() if v.requires_grad)})
    return model.cuda()


def optimizer_for(model,plan):
    import torch
    hp=plan['training'];groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        backbone='backbone.dinov3.' in name;no_decay=p.ndim<=1 or 'norm' in name or '.bn' in name
        key=(backbone,no_decay);groups.setdefault(key,[]).append(p)
    return torch.optim.AdamW([{'params':ps,'lr':hp['lr_backbone'] if b else hp['lr_head'],
        'base_lr':hp['lr_backbone'] if b else hp['lr_head'],'weight_decay':0. if nd else hp['weight_decay']}
        for (b,nd),ps in groups.items()],betas=(.9,.999))


def prediction(model,postprocessor,images,split,size,guard,filename):
    import torch
    import numpy as np
    from PIL import Image
    model.eval();postprocessor.eval();rows=[];started=time.monotonic()
    mean=torch.tensor([.485,.456,.406],device='cuda')[:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
    for i,r in enumerate(images):
        with Image.open(ROOT/'dataset/images'/split/r['file_name']) as img:
            assert img.size==(r['width'],r['height'])
            arr=np.asarray(img.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255.
        x=torch.from_numpy(arr.copy()).permute(2,0,1).cuda();x=((x-mean)/std)[None]
        with torch.inference_mode():
            pred=postprocessor(model(x),torch.tensor([[r['width'],r['height']]],device='cuda'))[0]
        for label,score,box in zip(pred['labels'].cpu().tolist(),pred['scores'].cpu().tolist(),pred['boxes'].cpu().tolist()):
            if score<.001:continue
            x1,y1,x2,y2=box;x1=min(max(x1,0),r['width']);x2=min(max(x2,0),r['width'])
            y1=min(max(y1,0),r['height']);y2=min(max(y2,0),r['height'])
            if x2<=x1 or y2<=y1:continue
            rounded=[round(float(v),3) for v in [x1,y1,x2-x1,y2-y1]]
            rounded[2]=round(min(rounded[2],r['width']-rounded[0]),3)
            rounded[3]=round(min(rounded[3],r['height']-rounded[1]),3)
            if min(rounded[2:])<=0:continue  # Quantization can collapse a tiny clipped box.
            assert np.isfinite([*rounded,score]).all()
            rows.append({'image_id':i+1,'category_id':int(label),'score':round(float(score),6),'bbox':rounded})
        guard.check()
        if i%25==0:dump(OUT/'progress.json',{'stage':'validation','images':i+1,'total':len(images),
            'peak_gpu_mib':guard.peak,'unix':time.time()})
    target=OUT/filename;dump(target,rows)
    metric=target.with_suffix('.metrics.json')
    subprocess.run([sys.executable,str(ROOT/'deimv2_canonical_eval.py'),'--annotation',str(ROOT/'dataset/annotations/instances_val.json'),
        '--predictions',str(target),'--output',str(metric)],check=True,timeout=240)
    result=json.loads(metric.read_text());result['inference_seconds']=time.monotonic()-started
    return result


def save_state(model,ema,optimizer,epoch,counters,best_ap,history,plan):
    import torch
    recovery={'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
        'epoch':epoch,'counters':counters,'best_ap':best_ap,'history':history,
        'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_python':random.getstate(),
        'rng_numpy':__import__('numpy').random.get_state(),
        'protocol_sha256':file_sha(ROOT/'deim_protocol.json'),'training_plan':plan['training']}
    tmp=OUT/'recovery.pth.staging';torch.save(recovery,tmp)
    previous=OUT/'recovery_receipt.json';ack=OUT/'local_recovery_ack.json'
    if previous.exists():
        prior=json.loads(previous.read_text());lease=json.loads((ROOT/'runtime_budget.json').read_text())
        while True:
            acknowledged=json.loads(ack.read_text()) if ack.exists() else {}
            if acknowledged.get('epoch')==prior['epoch'] and acknowledged.get('sha256')==prior['sha256']:break
            if time.time()>lease['pilot_deadline_unix']-90 and not (OUT/'PILOT_GATE.json').exists():
                raise TimeoutError('Previous recovery not acknowledged before pilot deadline')
            if time.time()>lease['continuation_deadline_unix']-90:raise TimeoutError('Recovery acknowledgement deadline')
            time.sleep(1)
    tmp.replace(OUT/'recovery.pth')
    immutable=OUT/f'recovery_epoch_{epoch:03d}.pth'
    os.link(OUT/'recovery.pth',immutable)
    if ack.exists():
        acknowledged=json.loads(ack.read_text())
        old=OUT/f"recovery_epoch_{acknowledged['epoch']:03d}.pth"
        if acknowledged['epoch']<epoch and old.exists() and file_sha(old)==acknowledged['sha256']:
            old.unlink()  # Retire only our acknowledged, redundant recovery link.
    dump(OUT/'recovery_receipt.json',{'epoch':epoch,'sha256':file_sha(OUT/'recovery.pth'),
        'file':immutable.name,'bytes':(OUT/'recovery.pth').stat().st_size,'optimizer_retained':True,'counters':counters,'unix':time.time()})


def run():
    import torch
    import numpy as np
    from engine.core import YAMLConfig
    from engine.optim import ModelEMA
    OUT.mkdir(exist_ok=True);plan=json.loads((ROOT/'deim_protocol.json').read_text());budget=json.loads((ROOT/'runtime_budget.json').read_text())
    assert not (OUT/'STARTED.json').exists()
    dump(OUT/'STARTED.json',{'unix':time.time(),'pid':os.getpid(),'protocol_sha256':file_sha(ROOT/'deim_protocol.json')})
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0)
    torch.cuda.set_per_process_memory_fraction(10.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(3407);random.seed(3407);np.random.seed(3407)
    source=original_annotations(plan);path=configure(plan);cfg=YAMLConfig(str(path));model=initialize(cfg,plan)
    criterion=cfg.criterion.cuda();postprocessor=cfg.postprocessor.cuda();guard=Guard(plan['budget']['peak_gpu_limit_mib'])
    hp=plan['training'];history=[];counters={'minibatches':0,'image_samples':0,'optimizer_steps':0};best_ap=-1.
    try:
        # Complete gradients and AdamW state allocation on the two densest training images.
        loader=cfg.train_dataloader;dataset=loader.dataset
        counts={r['id']:0 for r in source['train']['images']}
        for a in source['train']['annotations']:
            if not a.get('iscrowd',0):counts[a['image_id']]+=1
        indices=sorted(range(4500),key=lambda i:counts[source['train']['images'][i]['id']],reverse=True)[:2]
        dataset.set_epoch(hp['epochs']);preflight=[];chosen=None
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for batch in [1,2]:
            trial_ema=ModelEMA(model,decay=.9998,warmups=500).to('cuda')
            opt=optimizer_for(model,plan);model.train();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
            samples=targets=outputs=losses=loss=None
            try:
                samples,targets=loader.collate_fn([dataset[i] for i in indices[:batch]])
                samples=samples.cuda();targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
                with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(samples,targets=targets)
                with torch.autocast('cuda',enabled=False):losses=criterion(outputs,targets,epoch=0,step=0,global_step=0,epoch_step=4500//batch)
                loss=sum(losses.values());assert torch.isfinite(loss);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),hp['clip_max_norm'],error_if_nonfinite=True);opt.step()
                torch.cuda.synchronize();guard.check()
                preflight.append({'batch':batch,'seconds':time.monotonic()-tick,'allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'device_peak_mib':guard.peak,'max_training_annotations':max(counts.values()),'passed':True})
                chosen=batch
            except torch.cuda.OutOfMemoryError:
                preflight.append({'batch':batch,'passed':False,'reason':'Bounded allocator OOM; no parameter changes retained'})
            finally:
                opt.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)
                del samples,targets,outputs,losses,loss
                del opt,trial_ema;gc.collect();torch.cuda.empty_cache();model.load_state_dict(initial,strict=True)
            dump(OUT/'preflight.json',{'trials':preflight,'chosen_batch':chosen,'all_train_data_unchanged':True})
            if not preflight[-1]['passed']:break
        assert chosen is not None,'No batch passes the VRAM bound'
        del initial;del loader;cfg._train_dataloader=None
        cfg.yaml_cfg['train_dataloader']['total_batch_size']=chosen
        cfg.yaml_cfg['val_dataloader']['total_batch_size']=1
        torch.manual_seed(3407);random.seed(3407);np.random.seed(3407)
        train_loader=cfg.train_dataloader;optimizer=optimizer_for(model,plan);ema=ModelEMA(model,decay=.9998,warmups=500).to('cuda')
        batch=chosen;accum=hp['effective_batch']//batch;n_batches=len(train_loader);steps_per_epoch=math.ceil(n_batches/accum)
        dump(OUT/'runtime_environment.json',{'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,
            'gpu':torch.cuda.get_device_name(0),'actual_batch':batch,'gradient_accumulation':accum,'effective_batch':batch*accum,
            'training_images':4500,'validation_images':700,'validation_training':False,'precision':'bfloat16 train, float32 validation',
            'steps_per_epoch':steps_per_epoch,'protocol_sha256':file_sha(ROOT/'deim_protocol.json')})
        start=time.monotonic();pilot_pass=False;final_reason=None
        for epoch in range(hp['epochs']):
            deadline=budget['continuation_deadline_unix'] if pilot_pass else budget['pilot_deadline_unix']
            if time.time()>deadline-240:final_reason='Budget leaves insufficient validation/download time';break
            train_loader.set_epoch(epoch);model.train();criterion.train();optimizer.zero_grad(set_to_none=True);loss_total=0.;tick=time.monotonic()
            for i,(samples,targets) in enumerate(train_loader):
                if time.time()>deadline-180:
                    final_reason='Budget reached during epoch; last complete recovery preserved';break
                samples=samples.cuda(non_blocking=True);targets=[{k:v.cuda(non_blocking=True) for k,v in t.items()} for t in targets]
                group_start=(i//accum)*accum;divisor=min(accum,n_batches-group_start)
                progress=epoch+(i+1)/n_batches
                factor=min(1.,progress/hp['warmup_epochs'])*(.01+.99*(1+math.cos(math.pi*max(0.,progress-hp['warmup_epochs'])/(hp['epochs']-hp['warmup_epochs'])))/2)
                for pg in optimizer.param_groups:pg['lr']=pg['base_lr']*factor
                with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(samples,targets=targets)
                with torch.autocast('cuda',enabled=False):losses=criterion(outputs,targets,epoch=epoch,step=i,global_step=epoch*n_batches+i,epoch_step=n_batches)
                loss=sum(losses.values());assert torch.isfinite(loss),'Nonfinite loss; no automatic retry'
                (loss/divisor).backward();loss_total+=float(loss.detach());counters['minibatches']+=1;counters['image_samples']+=len(samples)
                if (i+1)%accum==0 or i+1==n_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),hp['clip_max_norm'],error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True);ema.update(model);counters['optimizer_steps']+=1
                guard.check()
                if i%20==0:dump(OUT/'progress.json',{'stage':'training','epoch':epoch+1,'batch':i+1,'batches':n_batches,
                    'counters':counters,'peak_gpu_mib':guard.peak,'loss':float(loss.detach()),'unix':time.time()})
            if final_reason:break
            train_seconds=time.monotonic()-tick
            # Preserve completed training before any inference/evaluation can fail.
            pending_prior=OUT/'pending_recovery_receipt.json'
            if pending_prior.exists():
                prior=json.loads(pending_prior.read_text());ack=OUT/'pending_recovery_ack.json'
                while True:
                    confirmed=json.loads(ack.read_text()) if ack.exists() else {}
                    if confirmed.get('epoch')==prior['epoch'] and confirmed.get('sha256')==prior['sha256']:break
                    if time.time()>deadline-90:raise TimeoutError('Pending recovery not acknowledged before deadline')
                    time.sleep(1)
                prior_path=OUT/prior['file']
                assert prior_path.name.startswith('pending_train_epoch_')
                prior_path.unlink()
            pending=OUT/f'pending_train_epoch_{epoch+1:03d}.pth'
            torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                'completed_train_epoch':epoch+1,'validation_pending':True,'counters':copy.deepcopy(counters),
                'best_ap':best_ap,'history':history,'training_plan':plan['training'],
                'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
                'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
                'protocol_sha256':file_sha(ROOT/'deim_protocol.json')},pending)
            dump(OUT/'pending_recovery_receipt.json',{'epoch':epoch+1,'file':pending.name,
                'sha256':file_sha(pending),'bytes':pending.stat().st_size,'validation_pending':True,
                'train_seconds':train_seconds,'counters':copy.deepcopy(counters),'unix':time.time()})
            del samples,targets,outputs,losses,loss;gc.collect();torch.cuda.empty_cache()
            metrics=prediction(ema.module,postprocessor,source['val']['images'],'val',1280,guard,'latest_valid_1280.json')
            ap=metrics['all_valid']['AP'];improved=ap>best_ap
            if improved:
                best_ap=ap;torch.save({'model':ema.module.state_dict(),'epoch':epoch+1,'config':cfg.yaml_cfg,
                    'protocol_sha256':file_sha(ROOT/'deim_protocol.json')},OUT/'best.pth')
                shutil.copy2(OUT/'latest_valid_1280.json',OUT/'best_valid_1280.json')
                dump(OUT/'best_metrics.json',{'epoch':epoch+1,**metrics,'checkpoint_sha256':file_sha(OUT/'best.pth')})
            record={'epoch':epoch+1,'train_seconds':train_seconds,'epoch_seconds':time.monotonic()-tick,'mean_loss':loss_total/n_batches,
                'metrics':metrics,'counters':copy.deepcopy(counters),'peak_gpu_mib':guard.peak,'unix':time.time()};history.append(record)
            dump(OUT/'last_epoch_metrics.json',record);dump(OUT/'history.json',history)
            save_state(model,ema,optimizer,epoch+1,counters,best_ap,history,plan)
            if not pilot_pass and epoch+1<hp['pilot_epochs']:
                needed=(hp['pilot_epochs']-len(history))*max(r['epoch_seconds'] for r in history)*1.10+180
                if time.time()+needed>budget['pilot_deadline_unix']:
                    final_reason='Measured runtime cannot complete the pilot within two units';break
            if epoch+1==hp['pilot_epochs']:
                projected=(hp['epochs']-len(history))*max(r['epoch_seconds'] for r in history)*1.15+300
                trend=history[-1]['metrics']['all_valid']['AP']-history[0]['metrics']['all_valid']['AP']
                checks={'AP_at_least_025':ap>=.25,'AP_small_at_least_015':metrics['all_valid']['small_AP']>=.15,
                    'AP_gain_since_epoch1_at_least_001':trend>=.01,'complete_schedule_fits_budget':time.time()+projected<budget['continuation_deadline_unix'],
                    'VRAM_within_limit':guard.peak<plan['budget']['peak_gpu_limit_mib']}
                pilot_pass=all(checks.values());dump(OUT/'PILOT_GATE.json',{'passed':pilot_pass,'checks':checks,
                    'projected_remaining_seconds_with_margin':projected,'protocol_sha256':file_sha(ROOT/'deim_protocol.json'),
                    'note':'Early screening heuristic, not proof of final architecture quality.','unix':time.time()})
                if not pilot_pass:final_reason='Predeclared pilot gate not met';break
            if pilot_pass and epoch+1>=15 and len(history)>=6:
                recent=[r['metrics']['all_valid']['AP'] for r in history[-6:]]
                if best_ap<.30 and max(recent)-min(recent)<.003:final_reason='Low AP and six-epoch plateau';break
        artifacts=['preflight.json','initialization.json','effective_config.yml','runtime_environment.json','history.json',
            'last_epoch_metrics.json','best_metrics.json','best.pth','best_valid_1280.json','recovery_receipt.json','recovery.pth','latest_valid_1280.json','latest_valid_1280.metrics.json']
        if (OUT/'PILOT_GATE.json').exists():artifacts.append('PILOT_GATE.json')
        artifacts=[n for n in artifacts if (OUT/n).exists()]
        dump(OUT/'READY.json',{'stage':'training_finished','completed_epochs':len(history),'planned_epochs':hp['epochs'],
            'reason':final_reason or 'Complete schedule finished','pilot_passed':pilot_pass,'best_AP':best_ap,
            'peak_gpu_mib':guard.peak,'worker_source_sha256':file_sha(Path(__file__)),'protocol_sha256':file_sha(ROOT/'deim_protocol.json'),
            'artifacts':{n:{'sha256':file_sha(OUT/n),'bytes':(OUT/n).stat().st_size} for n in artifacts},
            'validation_used_for_training':False,'test_images_used':0,'kaggle_submission':False,'unix':time.time()})
    finally:guard.done.set();guard.thread.join(timeout=2)


if __name__=='__main__':
    try:run()
    except Exception as e:
        OUT.mkdir(exist_ok=True);dump(OUT/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'unix':time.time()})
        raise
