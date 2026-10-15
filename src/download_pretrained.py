#!/usr/bin/env python3
"""Download only the two generic official pretrained models used by this work."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

SOURCES = {
    'yolo26l.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt',
        'sha256': '9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26',
    },
    'deim_model.safetensors': {
        'url': 'https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO/resolve/6ce90ce18c14271c8ce145c35626d9a6544f881b/model.safetensors',
        'sha256': 'c0d0f45fa62b785126db5b40f5530866fecaab2d340cea812d2ba5f0f5646163',
    },
}


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/pretrained'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        target = args.output / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.part')
            print('Downloading', name, flush=True)
            request = urllib.request.Request(source['url'], headers={'User-Agent': 'CVPDL-HW1-retraining'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('xb') as handle:
                shutil.copyfileobj(response, handle)
            if digest(temporary) != source['sha256']:
                raise ValueError(f'{name}: source checksum differs; retained .part for inspection')
            temporary.replace(target)
        if digest(target) != source['sha256']:
            raise ValueError(f'{name}: existing file checksum differs')
        print(name, source['sha256'], flush=True)
    (args.output / 'sources.json').write_text(json.dumps(SOURCES, indent=2) + '\n')


if __name__ == '__main__':
    main()
