"""Validation-only class-wise threshold/NMS selection; no training or test tuning."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

CONF_GRID = (.001, .005, .01, .03, .05, .10, .20)
NMS_GRID = (.70, .60, .50)  # Input detections already underwent NMS at <= .70.


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix(path.suffix + '.tmp')
    stage.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    stage.replace(path)


def file_sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def ground_truth(annotation):
    from pycocotools.coco import COCO
    original = json.loads(Path(annotation).read_text())
    images = [{**row, 'eval_id': index + 1} for index, row in enumerate(original['images'])]
    mapping = {row['id']: row['eval_id'] for row in images}
    data = deepcopy(original)
    data['images'] = [{**row, 'id': mapping[row['id']]} for row in data['images']]
    data['annotations'] = [{**row, 'image_id': mapping[row['image_id']]}
                           for row in data['annotations']]
    data.setdefault('info', {'description': 'HW1 validation-only postprocess'})
    gt = COCO()
    gt.dataset = data
    with contextlib.redirect_stdout(io.StringIO()): gt.createIndex()
    return gt, images


def validate_predictions(gt, predictions):
    for row in predictions:
        if row['image_id'] not in gt.imgs or row['category_id'] not in gt.cats:
            raise ValueError('Prediction image/class mapping does not match validation')
        x, y, w, h = row['bbox']
        image = gt.imgs[row['image_id']]
        if not np.isfinite([x, y, w, h, row['score']]).all() or not 0 <= row['score'] <= 1:
            raise ValueError('Invalid prediction numbers')
        if min(w, h) <= 0 or min(x, y) < -.01 or x+w > image['width']+.01 or y+h > image['height']+.01:
            raise ValueError('Prediction is outside image or has invalid area')


def apply_policy(predictions, config):
    grouped = defaultdict(list)
    for row in predictions:
        category = str(row['category_id'])
        if row['score'] >= config['confidence_by_class'][category]:
            grouped[row['image_id'], category].append(row)
    output = defaultdict(list)
    for (image, category), rows in grouped.items():
        rows = sorted(rows, key=lambda row: row['score'], reverse=True)
        boxes = np.array([row['bbox'] for row in rows], dtype=np.float64)
        boxes[:, 2:] += boxes[:, :2]
        areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
        alive = np.arange(len(rows))
        while alive.size:
            current = int(alive[0]); output[image].append(rows[current])
            rest = alive[1:]
            if not rest.size: break
            overlap = np.maximum(0, np.minimum(boxes[current, 2:], boxes[rest, 2:])
                                 - np.maximum(boxes[current, :2], boxes[rest, :2])).prod(axis=1)
            ious = overlap / np.maximum(areas[current]+areas[rest]-overlap, 1e-12)
            alive = rest[ious <= config['nms_iou_by_class'][category]]
    cap = config.get('max_det', 300)
    return [row for image in sorted(output)
            for row in sorted(output[image], key=lambda row: row['score'], reverse=True)[:cap]]


def baseline_config(categories, max_det=300):
    return {'confidence_by_class': {str(c): .001 for c in categories},
            'nms_iou_by_class': {str(c): .70 for c in categories}, 'max_det': max_det}


def evaluator(gt, predictions, image_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        if predictions:
            dt = gt.loadRes(deepcopy(predictions))
        else:
            dt = COCO()
            dt.dataset = {'images': list(gt.imgs.values()), 'categories': list(gt.cats.values()), 'annotations': []}
            dt.createIndex()
        result = COCOeval(gt, dt, 'bbox')
        result.params.imgIds = sorted(gt.imgs if image_ids is None else image_ids)
        result.params.maxDets = [1, 10, 100]  # Per image AND category; no extra evaluator global cap.
        result.evaluate()
    return result


def score(result, image_ids, thresholds=None):
    """Reuse official COCO matches for prefix confidence filtering, then accumulate.

    Removing a low-score suffix cannot change earlier score-greedy matches.
    Regression tests compare this optimization to fresh COCO evaluation.
    """
    original = result.evalImgs
    selected_ids = set(image_ids)
    filtered = []
    for record in original:
        if record is None or record['image_id'] not in selected_ids:
            filtered.append(None); continue
        if thresholds is not None:
            keep = np.array(record['dtScores']) >= thresholds[str(record['category_id'])]
            row = dict(record)
            row['dtScores'] = np.array(record['dtScores'])[keep].tolist()
            row['dtIds'] = np.array(record['dtIds'])[keep].tolist()
            row['dtMatches'] = record['dtMatches'][:, keep]
            row['dtIgnore'] = record['dtIgnore'][:, keep]
            filtered.append(row)
        else:
            filtered.append(record)
    result.evalImgs = filtered
    try:
        # COCOeval.accumulate indexes evalImgs by positional indices, not an
        # arbitrary new subset's IDs. Preserve its original image axis and mask
        # excluded records instead of silently evaluating the first N images.
        result.params.imgIds = list(result._paramsEval.imgIds)
        with contextlib.redirect_stdout(io.StringIO()): result.accumulate()
        precision = result.eval['precision']
        def mean(array):
            valid = array[array > -1]
            return float(valid.mean()) if valid.size else None
        return {'AP': mean(precision[:, :, :, 0, -1]),
                'AP50': mean(precision[0, :, :, 0, -1]),
                'small_AP': mean(precision[:, :, :, 1, -1]),
                'per_class_AP': {str(c): mean(precision[:, :, index, 0, -1])
                                 for index, c in enumerate(result.params.catIds)}}
    finally:
        result.evalImgs = original


def tuning_split(images):
    ordered = sorted(images, key=lambda row: hashlib.sha256(
        ('cvpdl-postprocess-3407:' + str(row['id'])).encode()).digest())
    half = len(ordered) // 2
    return sorted(row['eval_id'] for row in ordered[:half]), sorted(row['eval_id'] for row in ordered[half:])


def tune(annotation, predictions, output, source=None):
    gt, images = ground_truth(annotation)
    validate_predictions(gt, predictions)
    categories = sorted(gt.cats)
    calibration, audit = tuning_split(images)
    config = baseline_config(categories)
    best = {str(c): {'AP': -1., 'confidence': .001, 'iou': .70} for c in categories}
    grid = []
    for iou in NMS_GRID:
        candidate = baseline_config(categories)
        candidate['nms_iou_by_class'] = {str(c): iou for c in categories}
        pool = apply_policy(predictions, candidate)
        evaluation = evaluator(gt, pool)
        for confidence in CONF_GRID:
            thresholds = {str(c): confidence for c in categories}
            scores = score(evaluation, calibration, thresholds)
            grid.append({'iou': iou, 'confidence': confidence, 'calibration': scores})
            for category, ap in scores['per_class_AP'].items():
                # Prefer original NMS and lower threshold when AP ties.
                if ap is not None and ap > best[category]['AP'] + 1e-10:
                    best[category] = {'AP': ap, 'confidence': confidence, 'iou': iou}
        print(json.dumps({'nms_grid_completed': iou, 'images': len(images)}), flush=True)
    for category, choice in best.items():
        config['confidence_by_class'][category] = choice['confidence']
        config['nms_iou_by_class'][category] = choice['iou']
    baseline = apply_policy(predictions, baseline_config(categories))
    selected = apply_policy(predictions, config)
    base_eval, selected_eval = evaluator(gt, baseline), evaluator(gt, selected)
    comparisons = {}
    for name, ids in [('calibration', calibration), ('audit', audit), ('all_valid', sorted(gt.imgs))]:
        before, after = score(base_eval, ids), score(selected_eval, ids)
        comparisons[name] = {'baseline': before, 'selected': after, 'AP_delta': after['AP']-before['AP']}
    summary = {'method': 'Class-wise confidence and stricter NMS selected on calibration350 only',
               'training': False, 'test_labels_used': False, 'checkpoint_source': source,
               'annotation_sha256': file_sha(annotation), 'validation_images': len(images),
               'calibration_eval_ids': calibration, 'audit_eval_ids': audit,
               'audit_used_for_parameter_selection': False,
               'evaluator': {'name': 'COCOeval', 'iou': '.50:.95', 'maxDets': [1, 10, 100],
                             'cap_scope': 'per image/category', 'extra_global_eval_cap': None},
               'input_prediction_count': len(predictions), 'baseline_count': len(baseline),
               'selected_count': len(selected), 'postprocess_config': config,
               'per_class_selected': best, 'comparisons': comparisons, 'search_grid': grid,
               'scope': 'Audit is held out from postprocess selection, but model was already selected using full valid; not an independent model test.'}
    dump(Path(output)/'postprocess_selection.json', summary)
    dump(Path(output)/'postprocess_config.json', config)
    return summary


def read_submission(path):
    predictions, ids = [], []
    with Path(path).open(newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ['image_id', 'PredictionString']: raise ValueError('CSV header mismatch')
        for row in reader:
            image_id = row['image_id']; ids.append(image_id)
            fields = row['PredictionString'].split()
            if fields == ['none']: continue
            if len(fields) % 6: raise ValueError('Malformed boxes')
            for start in range(0, len(fields), 6):
                cls = int(fields[start]); conf, x, y, w, h = map(float, fields[start+1:start+6])
                predictions.append({'image_id': image_id, 'category_id': cls,
                                    'score': conf, 'bbox': [x, y, w, h]})
    if len(set(ids)) != len(ids): raise ValueError('Duplicate image IDs')
    return ids, predictions


def write_submission(path, ids, predictions):
    grouped = defaultdict(list)
    for row in predictions: grouped[row['image_id']].append(row)
    with Path(path).open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=['image_id', 'PredictionString'])
        writer.writeheader()
        for image_id in ids:
            fields = []
            for row in sorted(grouped[image_id], key=lambda r:r['score'], reverse=True):
                fields += [str(row['category_id']), f"{row['score']:.6f}"]
                fields += [f'{number:.3f}' for number in row['bbox']]
            writer.writerow({'image_id': image_id, 'PredictionString': ' '.join(fields) if fields else 'none'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--test-csv', type=Path)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text())
    summary = tune(args.annotation, predictions, args.out,
                   {'predictions_sha256': file_sha(args.predictions), 'weights': 'gentle_control public=.30215'})
    if args.test_csv:
        ids, test_predictions = read_submission(args.test_csv)
        selected = apply_policy(test_predictions, summary['postprocess_config'])
        output = args.out / 'gentle_control_1280_postprocess.csv'
        write_submission(output, ids, selected)
        dump(args.out/'test_transform.json', {'source': str(args.test_csv),
             'source_sha256': file_sha(args.test_csv), 'output_sha256': file_sha(output),
             'input_count': len(test_predictions), 'output_count': len(selected), 'images': len(ids),
             'input_nms_iou': .70, 'input_max_det': 300, 'cannot_recover_already_suppressed_boxes': True})
    print(json.dumps(summary['comparisons'], indent=2))
