#!/usr/bin/env python3
"""Prepare local YAML and numeric COCO IDs from the unmodified HW1 release."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/data'))
    args = parser.parse_args()
    import yaml
    root, out = args.data_root.resolve(), args.output.resolve()
    protocol = json.loads((Path(__file__).parent / 'records/deim_protocol.json').read_text())
    sources, hashes = {}, {}
    for split, count in [('train', 4500), ('val', 700)]:
        annotation = root / 'annotations' / f'instances_{split}.json'
        hashes[split] = sha(annotation)
        if hashes[split] != protocol['annotation_sha256'][split]:
            raise ValueError(f'{split}: use the original assignment annotation JSON')
        data = json.loads(annotation.read_text())
        if len(data['images']) != count or sorted(c['id'] for c in data['categories']) != list(range(10)):
            raise ValueError(f'{split}: unexpected image count or category IDs')
        for row in data['images']:
            name = row['file_name']
            if Path(name).name != name:
                raise ValueError('Expected flat assignment image directories')
            for path in [root / 'images' / split / name,
                         root / 'labels' / split / (Path(name).stem + '.txt')]:
                if not path.is_file():
                    raise FileNotFoundError(path)
        sources[split] = data
    if {r['id'] for r in sources['train']['images']} & {r['id'] for r in sources['val']['images']}:
        raise ValueError('Train and validation IDs overlap')
    out.mkdir(parents=True, exist_ok=False)
    for split, data in sources.items():
        numeric = copy.deepcopy(data)
        mapping = {r['id']: i + 1 for i, r in enumerate(data['images'])}
        for row in numeric['images']:
            row['id'] = mapping[row['id']]
        for i, row in enumerate(numeric['annotations']):
            row.update(id=i + 1, image_id=mapping[row['image_id']])
        (out / f'{split}_numeric.json').write_text(json.dumps(numeric) + '\n')
        (out / f'{split}_id_mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    config = {'path': str(root), 'train': 'images/train', 'val': 'images/val',
              'names': {c['id']: c['name'] for c in sources['train']['categories']}}
    (out / 'dataset.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (out / 'prepared.json').write_text(json.dumps({
        'data_root': str(root), 'annotation_sha256': hashes,
        'train_images': 4500, 'val_images': 700,
        'labels': 'Original supplied YOLO labels; no class remapping',
        'numeric_ids': '1-based IDs in original COCO image order',
        'test_accessed': False, 'original_dataset_modified': False,
    }, indent=2) + '\n')
    print(out / 'dataset.yaml')


if __name__ == '__main__':
    main()
