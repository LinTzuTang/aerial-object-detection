"""Resume retained DEIM states; canonical validation controls explicit early stopping."""
import copy,gc,json,math,os,shutil,time,traceback
from pathlib import Path
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
import deimv2_revision_remote as recipe
import deimv2_l_remote as base
from postprocess_policy import dump,file_sha
ROOT=base.ROOT;OUT=base.OUT


def host_memory():
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    status={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:'))}
    info={**mem,**status}
    for key,name in [('cgroup_current','memory.current'),('cgroup_limit','memory.max')]:
        p=Path('/sys/fs/cgroup')/name
        if p.exists():
            value=p.read_text().strip()
            if value!='max':info[key]=int(value)
    return info


def host_check():
    info=host_memory()
    if info.get('MemAvailable',10**12)<512*1024**2:raise MemoryError('Host RAM safety reserve reached; verified recovery retained')
    if info.get('cgroup_limit',10**15)-info.get('cgroup_current',0)<512*1024**2:raise MemoryError('Host cgroup safety reserve reached')
    return info


def run():
    import numpy as np
    import torch
    from engine.core import YAMLConfig
    from engine.optim import ModelEMA
    OUT.mkdir(exist_ok=True);assert not (OUT/'STARTED.json').exists()
    plan=json.loads((ROOT/'deim_protocol.json').read_text());hp=plan['training']
    budget=json.loads((ROOT/'runtime_budget.json').read_text());deadline=budget['total_hard_deadline_unix']
    dump(OUT/'STARTED.json',{'pid':os.getpid(),'unix':time.time(),'protocol_sha256':file_sha(ROOT/'deim_protocol.json')})
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0)
    torch.cuda.set_per_process_memory_fraction(10.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    source=base.original_annotations(plan);cfg=YAMLConfig(str(recipe.configure(plan)))
    cfg.yaml_cfg['train_dataloader']['total_batch_size']=2
    cfg.yaml_cfg['train_dataloader']['num_workers']=0
    checkpoint_path=ROOT/'resume.pth';assert file_sha(checkpoint_path)==plan['resume_sha256']
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    partial_batch=checkpoint.get('partial_epoch_batch')
    assert partial_batch==plan.get('resume_partial_epoch_batch')
    start_epoch=checkpoint['completed_epoch'];assert start_epoch==plan['resume_completed_whole_epochs'] and start_epoch<hp['epochs']
    model=cfg.model.cuda();model.load_state_dict(checkpoint['model'],strict=True)
    optimizer=recipe.optimizer_for(model,plan);optimizer.load_state_dict(checkpoint['optimizer'])
    # Continue the saved cosine, including its original group base LRs.
    assert all('base_lr' in pg for pg in optimizer.param_groups)
    elapsed_passes=hp['schedule_elapsed_passes'];schedule_passes=hp['schedule_total_passes']
    ema=ModelEMA(model,decay=.9998,warmups=500).to('cuda');ema.load_state_dict(checkpoint['ema'])
    counters=copy.deepcopy(checkpoint['counters'])
    torch.set_rng_state(checkpoint['rng_torch']);torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    __import__('random').setstate(checkpoint['rng_python']);np.random.set_state(checkpoint['rng_numpy'])
    del checkpoint;gc.collect()
    criterion=cfg.criterion.cuda();postprocessor=cfg.postprocessor.cuda();loader=cfg.train_dataloader
    guard=base.Guard(plan['budget']['peak_gpu_limit_mib']);history=copy.deepcopy(plan.get('resume_history',[]));sequence=0
    dump(OUT/'runtime_environment.json',{'torch':torch.__version__,'data_workers':0,'USE_TF':os.environ['USE_TF'],'host_memory':host_check()})
    resume_metrics=recipe.evaluate(ema.module,postprocessor,source['val']['images'],guard,'resume_baseline_valid.json')
    initial_ap=resume_metrics['all_valid']['AP'];early=plan['resume_early_stop_state']
    best_ap=early['best_AP'];anchor_ap=early['anchor_AP']
    resume_epoch=start_epoch+(partial_batch or 0)/2250
    best_epoch=early['best_epoch'];bad=early['patience_count'];reason='maximum_epochs';best_path=OUT/'best_ema.pth'
    if initial_ap>best_ap:
        best_ap=initial_ap;best_epoch=resume_epoch
        shutil.copy2(OUT/'resume_baseline_valid.json',OUT/'best_valid.json')
        shutil.copy2(OUT/'resume_baseline_valid.metrics.json',OUT/'best_valid.metrics.json')
        torch.save({'model':ema.module.state_dict(),'config':cfg.yaml_cfg,'epoch':best_epoch},best_path)
    else:
        historical=ROOT/'historical_best.pth';assert file_sha(historical)==plan['historical_best_sha256']
        retained=copy.deepcopy(ema.module.state_dict())
        old=torch.load(historical,map_location='cpu',weights_only=False)
        ema.module.load_state_dict(old['model'],strict=True);del old
        old_metrics=recipe.evaluate(ema.module,postprocessor,source['val']['images'],guard,'best_valid.json')
        assert abs(old_metrics['all_valid']['AP']-best_ap)<1e-5,'Historical best validation changed'
        shutil.copy2(historical,best_path);ema.module.load_state_dict(retained,strict=True);del retained
    # Partial-prefix revalidation does not reset or consume whole-epoch patience.
    dump(OUT/'history.json',history)
    dump(OUT/'EARLY_STOP_STATE.json',{**early,'best_epoch':best_epoch,'best_AP':best_ap})
    dump(OUT/'effective_start.json',{'resume_sha256':plan['resume_sha256'],'start_epoch':start_epoch,
        'initial_AP':initial_ap,'resume_metrics':resume_metrics,'partial_prefix_batches':partial_batch,
        'resumption':'Begin a new shuffled full pass after saved partial prefix; actual counters include extra prefix; not exact remaining-image replay',
        'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,'data_workers':0,'optimizer_moments_preserved':True,'initial_lrs':[g['lr'] for g in optimizer.param_groups],
        'schedule_base_lrs':[g['base_lr'] for g in optimizer.param_groups],'schedule_elapsed_passes':elapsed_passes,'schedule_total_passes':schedule_passes,
        'preserved_anchor_AP':anchor_ap,'preserved_patience_count':bad,
        'patience':hp['patience'],'min_delta':hp['min_delta'],'minimum_total_epochs':hp['minimum_total_epochs'],
        'validation_gradient_updates':False})
    try:
        for epoch in range(start_epoch,hp['epochs']):
            required=max([1583]+[r['epoch_seconds'] for r in history])+240
            if time.time()+required>deadline:reason='budget_before_next_epoch';break
            loader.set_epoch(epoch);model.train();criterion.train();optimizer.zero_grad(set_to_none=True)
            tick=time.monotonic();n=len(loader);loss_total=0.;start_steps=counters['optimizer_steps']
            for i,(samples,targets) in enumerate(loader):
                if time.time()>deadline-180:raise TimeoutError('Training deadline; retain last verified recovery')
                progress=min(1.,(elapsed_passes+epoch-start_epoch+(i+1)/n)/schedule_passes)
                factor=hp['final_lr_factor']+(1-hp['final_lr_factor'])*(1+math.cos(math.pi*progress))/2
                for pg in optimizer.param_groups:pg['lr']=pg['base_lr']*factor
                samples=samples.cuda(non_blocking=True);targets=[{k:v.cuda(non_blocking=True) for k,v in t.items()} for t in targets]
                with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(samples,targets=targets)
                with torch.autocast('cuda',enabled=False):losses=criterion(outputs,targets,epoch=epoch,step=i,global_step=epoch*n+i,epoch_step=n)
                loss=sum(losses.values());assert torch.isfinite(loss)
                (loss/min(8,n-(i//8)*8)).backward();loss_total+=float(loss.detach())
                counters['minibatches']+=1;counters['image_samples']+=len(samples)
                if (i+1)%8==0 or i+1==n:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),hp['clip_max_norm'],error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True);ema.update(model);counters['optimizer_steps']+=1
                    if counters['optimizer_steps']-start_steps in (128,256):
                        sequence+=1;recipe.snapshot(model,ema,optimizer,'earlystop',epoch,counters,sequence,plan,partial_batch=i+1)
                guard.check()
                if i%40==0:dump(OUT/'progress.json',{'stage':'training','arm':'earlystop','epoch':epoch+1,'batch':i+1,'batches':n,'counters':counters,'peak_gpu_mib':guard.peak,'host_memory':host_check(),'unix':time.time()})
            assert counters['optimizer_steps']-start_steps==282
            sequence+=1;recipe.snapshot(model,ema,optimizer,'earlystop',epoch+1,counters,sequence,plan)
            del samples,targets,outputs,losses,loss;gc.collect();torch.cuda.empty_cache()
            name=f'earlystop_epoch{epoch+1}_valid_original.json'
            metrics=recipe.evaluate(ema.module,postprocessor,source['val']['images'],guard,name)
            ap=metrics['all_valid']['AP']
            if ap>best_ap:
                best_ap=ap;best_epoch=epoch+1
                torch.save({'model':ema.module.state_dict(),'config':cfg.yaml_cfg,'epoch':best_epoch,'protocol_sha256':file_sha(ROOT/'deim_protocol.json')},best_path)
                shutil.copy2(OUT/name,OUT/'best_valid.json');shutil.copy2(OUT/name.replace('.json','.metrics.json'),OUT/'best_valid.metrics.json')
            if ap>=anchor_ap+hp['min_delta']:anchor_ap=ap;bad=0
            else:bad+=1
            record={'epoch':epoch+1,'metrics':metrics,'epoch_seconds':time.monotonic()-tick,'mean_loss':loss_total/n,'optimizer_equivalent_epochs':counters['optimizer_steps']/282,
                'completed_whole_epochs_excludes_prior_partial_prefix':epoch+1,
                'counters':copy.deepcopy(counters),'best_epoch':best_epoch,'best_AP':best_ap,'patience_count':bad,'peak_gpu_mib':guard.peak}
            history.append(record);dump(OUT/'history.json',history)
            dump(OUT/'EARLY_STOP_STATE.json',{'epoch':epoch+1,'best_epoch':best_epoch,'best_AP':best_ap,'patience_count':bad,'anchor_AP':anchor_ap,'minimum_total_epochs':hp['minimum_total_epochs']})
            if epoch+1>=hp['minimum_total_epochs'] and bad>=hp['patience']:reason='early_stopping';break
        decision={'stop_reason':reason,'early_stopping_reached':reason=='early_stopping','completed_epoch':history[-1]['epoch'] if history else start_epoch,
            'best_epoch':best_epoch,'best_AP':best_ap,'history':history,'resume_metrics':resume_metrics,
            'extra_partial_prefix_batches':plan.get('historical_partial_prefix_batches',0)+(partial_batch or 0),'actual_counters':counters,'valid_training':False,'test_images_used':0}
        dump(OUT/'DECISION.json',decision)
        names=['runtime_environment.json','effective_config.yml','effective_start.json','resume_baseline_valid.json','resume_baseline_valid.metrics.json','optimizer_groups.json','history.json','EARLY_STOP_STATE.json','DECISION.json','best_ema.pth','best_valid.json','best_valid.metrics.json','snapshot.json']
        names += [p.name for p in OUT.glob('earlystop_epoch*_valid_original.metrics.json')]
        if sequence:names.append(f'recovery_{sequence:03d}.pth')
        names=sorted(set(n for n in names if (OUT/n).exists()))
        dump(OUT/'READY.json',{'unix':time.time(),'decisions':decision,'completed_training_epochs':len(history)-len(plan.get('resume_history',[])),'peak_gpu_mib':guard.peak,
            'worker_source_sha256':file_sha(Path(__file__)),'protocol_sha256':file_sha(ROOT/'deim_protocol.json'),
            'artifacts':{n:{'bytes':(OUT/n).stat().st_size,'sha256':file_sha(OUT/n)} for n in names}})
    finally:guard.done.set();guard.thread.join(timeout=2)

if __name__=='__main__':
    try:run()
    except Exception as exc:
        OUT.mkdir(exist_ok=True);dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'unix':time.time()});raise
