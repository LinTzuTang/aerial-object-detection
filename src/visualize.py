#!/usr/bin/env python3
"""Draw saved detections or preview the project's fixed CLAHE policy on CPU.

No detector is loaded. Prediction files and source images are never modified.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from prediction_io import parse

NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
         'tricycle', 'awning-tricycle', 'bus', 'motor']
COLORS = ['#54a8ff', '#ff9f43', '#fa78d2', '#79d6d0', '#92c95c',
          '#f1ce54', '#c393ff', '#ff7777', '#b7c5fc', '#e7a273']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def read_json(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def render(source, rows, title, confidence=.25, boxes_only=False):
    image = source.convert('RGB')
    header = 66
    canvas = Image.new('RGB', (image.width, image.height + header), '#17212b')
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    chosen = [r for r in rows if r.get('score', 1.) >= confidence]
    draw.text((12, 8), title, font=font(18), fill='white')
    draw.text((12, 34), f'{len(chosen)} boxes | threshold {confidence:g} | no display cap',
              font=font(15), fill='white')
    label_font = font(max(14, min(22, round(image.width / 90))))
    occupied = []
    for row in sorted(chosen, key=lambda r: r.get('score', 1.)):
        category = row['category_id']
        score = row.get('score', 1.)
        values = [score, *row['bbox']]
        if category not in range(10) or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid category or nonfinite detection')
        x, y, w, h = row['bbox']
        if not 0 <= score <= 1 or w <= 0 or h <= 0:
            raise ValueError('Invalid score or box dimensions')
        # Clip the display to the canvas; do not change the saved predictions.
        x1, y1 = max(0., x), max(0., y)
        x2, y2 = min(image.width - 1., x + w), min(image.height - 1., y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        color = COLORS[category]
        draw.rectangle((x1, y1 + header, x2, y2 + header), outline=color, width=2)
        if boxes_only:
            continue
        label = NAMES[category] + (f' {score:.2f}' if 'score' in row else '')
        bounds = draw.textbbox((0, 0), label, font=label_font)
        tw, th = bounds[2] - bounds[0] + 6, bounds[3] - bounds[1] + 4
        left, top = max(0., min(x1, canvas.width - tw)), max(header, y1 + header - th)
        choices = []
        for dy in [0, -1, 1, -2, 2, -3, 3]:
            for dx in [0, -tw, tw]:
                lx = max(0., min(left + dx, canvas.width - tw))
                ly = max(header, min(top + dy * (th + 2), canvas.height - th))
                box = (lx, ly, lx + tw, ly + th)
                overlap = sum(max(0., min(box[2], o[2]) - max(box[0], o[0])) *
                              max(0., min(box[3], o[3]) - max(box[1], o[1])) for o in occupied)
                choices.append((overlap, abs(lx - left) + abs(ly - top), lx, ly))
        _, distance, lx, ly = min(choices)
        if distance > th:
            draw.line(((x1 + x2) / 2, y1 + header, lx + tw / 2, ly + th / 2), fill=color)
        occupied.append((lx, ly, lx + tw, ly + th))
        draw.rectangle(occupied[-1], fill='#17212b')
        draw.text((lx + 3, ly + 2 - bounds[1]), label, fill=color, font=label_font)
    return canvas


def draw_predictions(args):
    requested = args.image_ids
    if len(set(requested)) != len(requested):
        raise ValueError('Duplicate --image-ids')
    for stem in requested:
        if Path(stem).name != stem or stem in ('.', '..'):
            raise ValueError('Use flat image stems without directories')
    annotation = read_json(args.annotation) if args.annotation else None
    image_meta, numeric_ids, original_ids, ground_truth = {}, {}, {}, defaultdict(list)
    if annotation:
        for index, record in enumerate(annotation['images'], 1):
            if Path(record['file_name']).name != record['file_name']:
                raise ValueError('Expected flat image filenames')
            stem = Path(record['file_name']).stem
            if stem in image_meta:
                raise ValueError('Duplicate image stem in annotations')
            image_meta[stem] = record
            numeric_ids[index] = stem
            original_ids[record['id']] = stem
        for row in annotation['annotations']:
            ground_truth[original_ids[row['image_id']]].append(row)
        for stem in requested:
            if stem not in image_meta:
                raise ValueError(f'Image absent from annotations: {stem}')
    groups = {stem: [] for stem in requested}
    if args.predictions.suffix.lower() == '.csv':
        found = set()
        with args.predictions.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'image_id', 'PredictionString'}.issubset(reader.fieldnames or []):
                raise ValueError('Expected image_id,PredictionString columns')
            for row in reader:
                stem = row['image_id']
                if stem in groups:
                    if stem in found:
                        raise ValueError(f'Duplicate CSV image: {stem}')
                    groups[stem] = parse(row['PredictionString'])
                    found.add(stem)
        if set(requested) != found:
            raise ValueError(f'Images missing from CSV: {sorted(set(requested) - found)}')
    else:
        if not annotation:
            raise ValueError('COCO JSON predictions require --annotation to map image IDs')
        id_map = numeric_ids if args.id_space == 'numeric' else original_ids
        for row in read_json(args.predictions):
            if row['image_id'] not in id_map:
                raise ValueError('Unknown prediction image ID; check --id-space')
            stem = id_map[row['image_id']]
            if stem in groups:
                groups[stem].append(row)
    paths = {stem: args.images / (image_meta[stem]['file_name'] if annotation else stem + '.jpg')
             for stem in requested}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_sha = sha(args.predictions)
    records = []
    for stem, path in paths.items():
        with Image.open(path) as handle:
            source = handle.convert('RGB')
        pred = render(source, groups[stem], stem + ' | predictions', args.confidence, args.boxes_only)
        if annotation:
            gt = render(source, ground_truth[stem], stem + ' | ground truth', 0, args.boxes_only)
            panel = Image.new('RGB', (gt.width + pred.width, gt.height))
            panel.paste(gt, (0, 0))
            panel.paste(pred, (gt.width, 0))
        else:
            panel = pred
        output = args.output_dir / (stem + '.jpg')
        panel.save(output, quality=95)
        records.append({'image_id': stem, 'source_image_sha256': sha(path),
                        'prediction_sha256': source_sha, 'confidence': args.confidence,
                        'predictions_total': len(groups[stem]),
                        'predictions_above_threshold': sum(r['score'] >= args.confidence for r in groups[stem]),
                        'figure': output.name, 'figure_sha256': sha(output)})
    (args.output_dir / 'provenance.json').write_text(json.dumps({
        'images': records, 'annotation_sha256': sha(args.annotation) if annotation else None,
        'json_id_space': args.id_space, 'display_cap': None,
        'note': 'Display only. Source predictions and evaluation metrics are not changed.'}, indent=2) + '\n')
    print(args.output_dir)


def preview_clahe(args):
    from luminance_clahe import POLICY, preprocess_pil
    if args.output.suffix.lower() != '.png':
        raise ValueError('Use a .png output for the comparison panel')
    metadata = args.output.with_suffix('.json')
    if args.output.exists() or metadata.exists():
        raise FileExistsError('Use a new output path')
    with Image.open(args.image) as handle:
        original = handle.convert('RGB')
    processed = preprocess_pil(original)
    panel = Image.new('RGB', (original.width * 2, original.height + 42), '#17212b')
    panel.paste(original, (0, 42))
    panel.paste(processed, (original.width, 42))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), 'Original', font=font(18), fill='white')
    draw.text((original.width + 12, 10), 'CLAHE | clip 2 | grid 8x8 | blend 0.5', font=font(18), fill='white')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(args.output)
    metadata.write_text(json.dumps({'policy': POLICY, 'source_sha256': sha(args.image),
        'output_sha256': sha(args.output), 'source_modified': False}, indent=2) + '\n')
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pred = commands.add_parser('predictions', help='Draw CSV or COCO JSON detections; optional GT comparison')
    pred.add_argument('--images', type=Path, required=True)
    pred.add_argument('--predictions', type=Path, required=True)
    pred.add_argument('--image-ids', nargs='+', required=True, help='Image filename stems, without .jpg')
    pred.add_argument('--annotation', type=Path, help='Original COCO annotation JSON; adds a GT panel')
    pred.add_argument('--id-space', choices=['numeric', 'original'], default='numeric',
                      help='JSON prediction IDs: numeric=1-based annotation order; original=annotation IDs')
    pred.add_argument('--confidence', type=float, default=.25)
    pred.add_argument('--boxes-only', action='store_true', help='Hide text labels in dense scenes')
    pred.add_argument('--output-dir', type=Path, required=True)
    pred.set_defaults(run=draw_predictions)
    clahe = commands.add_parser('clahe', help='Compare original image with the fixed training/inference preprocessing')
    clahe.add_argument('--image', type=Path, required=True)
    clahe.add_argument('--output', type=Path, required=True)
    clahe.set_defaults(run=preview_clahe)
    args = parser.parse_args()
    if args.command == 'predictions' and not 0 <= args.confidence <= 1:
        parser.error('--confidence must be between 0 and 1')
    args.run(args)


if __name__ == '__main__':
    main()
