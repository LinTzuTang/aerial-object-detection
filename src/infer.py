"""Reproduce a recorded validation-frozen member recipe, one model at a time."""
import argparse,csv,gc,hashlib,json,sys
from pathlib import Path
from prediction_io import serialize
from fuse import fuse_csvs
from yolo_predict import predict_batch
CODE=Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def deim_member(spec,images,ids,target,device):
    import os
    os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    import numpy as np,torch,yaml
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    from PIL import Image
    sys.path.insert(0,str(CODE/'vendor/DEIMv2'))
    from engine.core import YAMLConfig
    checkpoint=torch.load(CODE/spec['checkpoint'],map_location='cpu',weights_only=False)
    config=target.with_suffix('.yml');config.write_text(yaml.safe_dump(checkpoint['config'],sort_keys=False))
    cfg=YAMLConfig(str(config));model=cfg.model.to(device);model.load_state_dict(checkpoint['model'],strict=True)
    postprocessor=cfg.postprocessor.to(device);model.eval();postprocessor.eval();del checkpoint
    mean=torch.tensor([.485,.456,.406],device=device)[:,None,None];std=torch.tensor([.229,.224,.225],device=device)[:,None,None]
    with target.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['image_id','PredictionString'])
        for n,stem in enumerate(ids,1):
            with Image.open(Path(images)/(stem+'.jpg')) as image:
                width,height=image.size
                x=np.asarray(image.convert('RGB').resize((1280,1280),Image.Resampling.BILINEAR),dtype=np.float32)/255.
            x=torch.from_numpy(x.copy()).permute(2,0,1).to(device);x=((x-mean)/std)[None]
            with torch.inference_mode():pred=postprocessor(model(x),torch.tensor([[width,height]],device=device))[0]
            rows=[]
            for cls,score,box in zip(pred['labels'].cpu().tolist(),pred['scores'].cpu().tolist(),pred['boxes'].cpu().tolist()):
                if score<.001:continue
                x1,y1,x2,y2=box;x1,x2=max(0.,min(x1,width)),max(0.,min(x2,width));y1,y2=max(0.,min(y1,height)),max(0.,min(y2,height))
                bbox=[round(float(v),3) for v in (x1,y1,x2-x1,y2-y1)]
                bbox[2]=round(min(bbox[2],width-bbox[0]),3);bbox[3]=round(min(bbox[3],height-bbox[1]),3)
                if min(bbox[2:])>0:rows.append({'category_id':int(cls),'score':round(float(score),6),'bbox':bbox})
            if spec.get('nms',False):
                from postprocess_policy import apply_policy,baseline_config
                rows=apply_policy([{**r,'image_id':stem} for r in rows],baseline_config(range(10)))
            writer.writerow([stem,serialize(rows)])
            if n%100==0:print(f"{spec['key']}: {n}/1800",flush=True)
    del model,postprocessor,cfg;gc.collect()


def main():
    p=argparse.ArgumentParser();p.add_argument('--images',required=True);p.add_argument('--sample',required=True)
    p.add_argument('--recipe',default=str(CODE.parent/'configs/final_ensemble.json'));p.add_argument('--workdir',default='frozen_members_reproduced')
    p.add_argument('--output',default='submission_reproduced.csv');p.add_argument('--device',default='0');a=p.parse_args()
    import torch,ultralytics
    from ultralytics import YOLO
    from runtime_guard import start_guard
    if ultralytics.__version__!='8.4.173':raise ValueError('Install pinned requirements')
    recipe=json.loads(Path(a.recipe).read_text())
    if digest(a.sample)!=recipe['sample_sha256']:raise ValueError('Unexpected sample file')
    with Path(a.sample).open(newline='') as f:ids=[r['image_id'] for r in csv.DictReader(f)]
    if len(ids)!=1800 or len(set(ids))!=1800:raise ValueError('Expected1800 unique images')
    work=Path(a.workdir);work.mkdir(parents=True,exist_ok=False)
    if Path(a.output).exists():raise FileExistsError(a.output)
    start_guard(a.device);members=[];device=torch.device('cuda:'+a.device if a.device.isdigit() else a.device)
    for spec in recipe['members']:
        checkpoint=CODE/spec['checkpoint']
        if digest(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch: '+spec['key'])
        target=work/(spec['key']+'.csv');members.append(target)
        if spec.get('kind','yolo')=='deim':deim_member(spec,a.images,ids,target,device)
        else:
            from luminance_clahe import preprocess
            model=YOLO(str(checkpoint));model.overrides['half']=False
            transform=(lambda image:preprocess(image,color_order='BGR')) if spec.get('processed',False) else ((lambda image:image) if spec.get('array_inputs',False) else None)
            with target.open('x',newline='') as f:
                writer=csv.writer(f);writer.writerow(['image_id','PredictionString'])
                for start in range(0,len(ids),2):
                    selected=ids[start:start+2]
                    groups=predict_batch(model,[Path(a.images)/(stem+'.jpg') for stem in selected],imgsz=spec['size'],
                        conf=.001,iou=.7,max_det=300,device=a.device,transform=transform,flip=spec.get('flip',True))
                    if model.predictor.model.end2end or model.predictor.model.fp16:raise RuntimeError('Expected FP32 one-to-many NMS backend')
                    for stem,group in zip(selected,groups):
                        rows=[{'category_id':c,'score':s,'bbox':[x1,y1,x2-x1,y2-y1]} for c,s,x1,y1,x2,y2 in group]
                        writer.writerow([stem,serialize(rows)])
                    if (start+2)%100==0:print(f"{spec['key']}: {start+2}/1800",flush=True)
            del model;gc.collect()
        if a.device.isdigit():torch.cuda.empty_cache()
        elif a.device=='mps':torch.mps.empty_cache()
    sha=fuse_csvs(members,recipe['weights'],a.images,a.sample,a.output,recipe.get('iou',.75))
    print(json.dumps({'output_sha256':sha,'reference_sha256':recipe.get('submission_sha256'),
        'byte_identical':sha==recipe.get('submission_sha256'),'note':'Original inference uses CUDA FP32; hardware changes can affect numerical results.'}))

if __name__=='__main__':main()
