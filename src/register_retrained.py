#!/usr/bin/env python3
"""Bind new training outputs to the fixed six-pool inference recipe."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

CODE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['gentle', 'fresh', 'mosaic', 'deim', 'clahe']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/retrained_deployment.json'))
    args = parser.parse_args()
    source = {'gentle': args.gentle, 'fresh': args.fresh, 'control': args.mosaic, 'deim': args.deim, 'clahe': args.clahe}
    names = {'gentle': 'best.pt', 'fresh': 'fresh_epoch37.pt',
             'control': 'mosaic03_epoch10.pt', 'deim': 'deim_best_ema.pth', 'clahe': 'clahe_epoch5.pt'}
    recipe = json.loads((CODE.parent / 'configs/final_ensemble.json').read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    for key, path in source.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        target = CODE.parent / 'checkpoints' / names[key]
        if target.exists():
            raise FileExistsError(f'Refusing to replace {target}; use a fresh unpacked copy')
    original = copy.deepcopy(recipe)
    provenance = {}
    for key, path in source.items():
        target = CODE.parent / 'checkpoints' / names[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = sha(target)
        if digest != sha(path):
            raise IOError('Checkpoint copy checksum mismatch')
        provenance[key] = {'source': str(path.resolve()), 'sha256': digest}
        for member in recipe['members']:
            if member['checkpoint_key'] == key:
                member['checkpoint_sha256'] = digest
    recipe['reference_submission_sha256'] = original['submission_sha256']
    recipe['submission_sha256'] = None
    recipe['retrained'] = True
    recipe['retraining_sources'] = provenance
    recipe['note'] = ('New weights, fixed inference settings. Historical report scores apply to the original '
                      'checkpoints; measure this retraining separately. No byte-identical CSV claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(recipe, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
