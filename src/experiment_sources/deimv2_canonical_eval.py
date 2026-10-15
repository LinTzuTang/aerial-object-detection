"""Canonical COCO evaluation in a process that never imports the DEIM engine."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from postprocess_policy import ground_truth, validate_predictions, evaluator, tuning_split, dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--annotation',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    gt,images=ground_truth(a.annotation);rows=json.loads(Path(a.predictions).read_text())
    validate_predictions(gt,rows);ev=evaluator(gt,rows);cal,audit=tuning_split(images);out={}
    original=ev.evalImgs
    for name,ids in [('calibration',cal),('audit',audit),('all_valid',sorted(gt.imgs))]:
        keep=set(ids);ev.evalImgs=[r if r is None or r['image_id'] in keep else None for r in original]
        ev.params.imgIds=list(ev._paramsEval.imgIds)
        with contextlib.redirect_stdout(io.StringIO()):ev.accumulate()
        pr=ev.eval['precision']
        def mean(v):
            v=v[v>-1];return float(v.mean()) if v.size else None
        out[name]={'AP':mean(pr[:,:,:,0,-1]),'AP50':mean(pr[0,:,:,0,-1]),'AP75':mean(pr[5,:,:,0,-1]),
            'small_AP':mean(pr[:,:,:,1,-1]),'per_class_AP':{str(c):mean(pr[:,:,i,0,-1]) for i,c in enumerate(ev.params.catIds)}}
    import importlib.metadata
    out['evaluator']={'package':'pycocotools','version':importlib.metadata.version('pycocotools'),
        'implementation':ev.__class__.__module__,'maxDets':[1,10,100],'prediction_global_cap':300,
        'note':'Validation already reused for model selection; audit is not an independent model test.'}
    assert out['evaluator']['implementation']=='pycocotools.cocoeval'
    dump(a.output,out);print(json.dumps(out['all_valid']),flush=True)


if __name__=='__main__':main()
