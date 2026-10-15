"""Stream any frozen WBF member list with recorded input/output precision."""
import argparse,csv,hashlib,json
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from PIL import Image
from prediction_io import parse,serialize
from vendor.ensemble_boxes_wbf import weighted_boxes_fusion


def fuse_rows(pools,width,height,weights,iou=.75):
    boxes=[];scores=[];labels=[]
    for pool in pools:
        boxes.append([[r['bbox'][0]/width,r['bbox'][1]/height,(r['bbox'][0]+r['bbox'][2])/width,(r['bbox'][1]+r['bbox'][3])/height] for r in pool])
        scores.append([r['score'] for r in pool]);labels.append([r['category_id'] for r in pool])
    b,s,c=weighted_boxes_fusion(boxes,scores,labels,weights=weights,iou_thr=iou,skip_box_thr=.001,conf_type='box_and_model_avg')
    result=[]
    for box,score,cls in zip(b[:300],s[:300],c[:300]):
        x1,y1,x2,y2=np.clip(box,0,1)*[width,height,width,height]
        if x2>x1 and y2>y1:result.append({'category_id':int(cls),'score':float(score),'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)]})
    return result


def fuse_csvs(members,weights,images,sample,output,iou=.75):
    if len(members)!=len(weights) or not members:raise ValueError('Member/weight mismatch')
    with ExitStack() as stack:
        readers=[csv.DictReader(stack.enter_context(Path(p).open(newline=''))) for p in members]
        expected=csv.DictReader(stack.enter_context(Path(sample).open(newline='')))
        writer=csv.writer(stack.enter_context(Path(output).open('x',newline='')));writer.writerow(['image_id','PredictionString']);seen=set()
        for n,r in enumerate(expected,1):
            stem=r['image_id'];pools=[]
            if stem in seen:raise ValueError('Duplicate image')
            seen.add(stem)
            for reader in readers:
                member=next(reader);assert member['image_id']==stem;pools.append(parse(member['PredictionString']))
            if len(pools)==1:rows=pools[0]
            else:
                with Image.open(Path(images)/(stem+'.jpg')) as image:w,h=image.size
                rows=fuse_rows(pools,w,h,weights,iou)
            writer.writerow([stem,serialize(rows)])
            if n%100==0:print(f'Frozen fusion {n}/1800',flush=True)
        if len(seen)!=1800 or any(next(r,None) is not None for r in readers):raise ValueError('Expected1800 aligned rows')
    with Path(output).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--members',nargs='+',required=True);p.add_argument('--weights',nargs='+',type=float,required=True)
    p.add_argument('--images',required=True);p.add_argument('--sample',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps({'sha256':fuse_csvs(a.members,a.weights,a.images,a.sample,a.output)}))
