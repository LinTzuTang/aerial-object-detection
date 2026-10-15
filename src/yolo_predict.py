#!/usr/bin/env python3
"""YOLO prediction helpers and a single-checkpoint inference entry point.

The released protocol is 1536-pixel inference with horizontal-flip TTA,
confidence 0.001, class-wise greedy NMS at IoU 0.70 and a 300-box image cap.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def nms_classwise(detections, iou_threshold=0.70, max_det=300):
    """detections: (class_id, confidence, x1, y1, x2, y2)."""
    kept = []
    by_class = defaultdict(list)
    for row in detections:
        by_class[int(row[0])].append(row)
    for rows in by_class.values():
        rows = sorted(rows, key=lambda r: r[1], reverse=True)
        # Match the recorded xywh -> xyxy arithmetic before NMS.
        boxes = np.asarray([[r[2],r[3],r[4]-r[2],r[5]-r[3]] for r in rows], dtype=np.float64)
        boxes[:, 2:] += boxes[:, :2]
        areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
        alive = np.arange(len(rows))
        while alive.size:
            index = int(alive[0])
            current = rows[index]
            kept.append(current)
            rest = alive[1:]
            if not rest.size: break
            overlap = np.maximum(0, np.minimum(boxes[index, 2:], boxes[rest, 2:])
                - np.maximum(boxes[index, :2], boxes[rest, :2])).prod(axis=1)
            ious = overlap / np.maximum(areas[index]+areas[rest]-overlap, 1e-12)
            alive = rest[ious <= iou_threshold]
    return sorted(kept, key=lambda r: r[1], reverse=True)[:max_det]


def predict_batch(model, paths, imgsz=1536, conf=.001, iou=.70, max_det=300,
                  device="0", transform=None, flip=True):
    """Match the recorded batch-two FP32 protocol, including rectangular padding."""
    frames=[cv2.imread(str(p)) for p in paths]
    if any(im is None for im in frames): raise FileNotFoundError(paths)
    processed=[transform(im) if transform else im for im in frames]
    # Keep file-path decoding for the unmodified original view, as in the release.
    views=[(False,processed if transform else [str(p) for p in paths])]
    if flip: views.append((True,[cv2.flip(im,1) for im in processed]))
    groups=[[] for _ in paths]
    for reflected, inputs in views:
        outputs=model.predict(inputs,imgsz=imgsz,conf=conf,iou=iou,max_det=max_det,
            batch=len(paths),device=device,verbose=False,rect=True,nms=None,half=False)
        for k,(result,frame) in enumerate(zip(outputs,frames)):
            h,w=frame.shape[:2]
            if result.boxes is None:continue
            for box,score,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.int().cpu().tolist()):
                x1,y1,x2,y2=box
                if reflected:x1,x2=w-x2,w-x1
                x1,x2=max(0.,x1),min(float(w),x2);y1,y2=max(0.,y1),min(float(h),y2)
                if x2>x1 and y2>y1:groups[k].append((int(cls),float(score),x1,y1,x2,y2))
    return [nms_classwise(rows,iou,max_det) for rows in groups]


def predict_image(model,path,imgsz=1536,conf=.001,iou=.70,max_det=300,device="0"):
    return predict_batch(model,[path],imgsz,conf,iou,max_det,device)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory containing test JPGs")
    parser.add_argument("--sample", required=True, help="Kaggle sample_submission.csv")
    parser.add_argument("--output", default="submission_test.csv")
    parser.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "checkpoints/best.pt"))
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2, help="Recorded inference batch; keep 2 for reproduction")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    args = parser.parse_args()
    from runtime_guard import start_guard
    start_guard(args.device)
    if args.batch < 1: raise ValueError("batch must be positive")
    from ultralytics import YOLO
    model = YOLO(args.weights)
    image_dir = Path(args.images)
    with open(args.sample, newline="", encoding="utf-8") as f:
        sample = list(csv.DictReader(f))
    if not sample or list(sample[0]) != ["image_id", "PredictionString"]:
        raise ValueError("Unexpected sample_submission.csv header")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "PredictionString"])
        for start in range(0,len(sample),args.batch):
            selected=sample[start:start+args.batch]
            groups=predict_batch(model,[image_dir/(r['image_id']+'.jpg') for r in selected],
                args.imgsz,args.conf,args.iou,args.max_det,args.device)
            for row,detections in zip(selected,groups):
                fields=[]
                for cls,score,x1,y1,x2,y2 in detections:
                    fields.extend((str(cls),f"{score:.6f}",f"{x1:.3f}",f"{y1:.3f}",f"{x2-x1:.3f}",f"{y2-y1:.3f}"))
                writer.writerow([row['image_id'],' '.join(fields) if fields else 'none'])
            if (start+len(selected))%100==0:print(f"{start+len(selected)}/{len(sample)}",flush=True)
    print(f"Wrote {len(sample)} rows to {output}")


if __name__ == "__main__":
    main()
