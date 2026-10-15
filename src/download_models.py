#!/usr/bin/env python3
"""Download and install the five original checkpoints from a verified archive."""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', help='Google Drive file share URL or a local model ZIP')
    args = parser.parse_args()
    code_dir = Path(__file__).resolve().parent
    config_path = code_dir / 'records/model_download.json'
    config = json.loads(config_path.read_text())
    destination = (code_dir / config['destination']).resolve()
    models = config['models']
    missing = []
    for name, spec in models.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid checkpoint filename in model_download.json')
        target = destination / name
        if target.is_symlink():
            raise FileExistsError(f'Expected a regular checkpoint file: {target}')
        if target.exists():
            if not target.is_file() or sha256(target) != spec['sha256']:
                raise FileExistsError(f'Existing file has a different hash; preserved: {target}')
        else:
            missing.append(name)
    if not missing:
        print(f'All five original checkpoints are present and verified: {destination}')
        return

    source = args.source or os.environ.get('CVPDL_MODEL_URL') or config.get('url')
    if not source:
        parser.error('Provide the uploaded ZIP share URL, or set "url" in ' + str(config_path))
    is_url = source.startswith(('https://', 'http://'))
    if is_url:
        parsed = urlsplit(source)
        if parsed.scheme != 'https' or parsed.hostname != 'drive.google.com' or '/folders/' in parsed.path:
            parser.error('Use an HTTPS Google Drive file share link, not a folder link')
        match = re.search(r'/file/d/([A-Za-z0-9_-]+)', parsed.path)
        file_id = match.group(1) if match else parse_qs(parsed.query).get('id', [''])[0]
        if not re.fullmatch(r'[A-Za-z0-9_-]+', file_id):
            parser.error('Could not identify the Google Drive file ID')
        try:
            import gdown
        except ImportError as error:
            raise RuntimeError('Install the downloader: python -m pip install gdown==5.2.0') from error
        local_archive = None
    else:
        local_archive = Path(source).expanduser().resolve()
        if not local_archive.is_file():
            parser.error('Source is neither an existing ZIP nor a Google Drive share URL')

    destination.parent.mkdir(parents=True, exist_ok=True)
    required = sum(models[name]['bytes'] for name in missing) + 64 * 1024**2
    if is_url:
        required += config['bytes']
    if shutil.disk_usage(destination.parent).free < required:
        raise RuntimeError(f'Need approximately {required / 1024**2:.0f} MiB free for download and extraction')

    # Stage on the destination filesystem. Only these temporary files are removed.
    with tempfile.TemporaryDirectory(prefix='.model-download-', dir=destination.parent) as scratch:
        stage = Path(scratch)
        archive = local_archive or stage / config['archive']
        if is_url:
            result = gdown.download(id=file_id, output=str(archive),
                                    use_cookies=False, verify=True, quiet=False)
            if not result:
                raise RuntimeError('Download failed; check that anyone with the link can download the file')
        if archive.stat().st_size != config['bytes'] or sha256(archive) != config['sha256']:
            raise ValueError('Model ZIP size/SHA-256 mismatch; no checkpoints installed')

        # Read only the named model members; never extract arbitrary ZIP paths.
        with zipfile.ZipFile(archive) as bundle:
            for name in missing:
                member = config['archive_prefix'] + 'weights/' + name
                if bundle.getinfo(member).file_size != models[name]['bytes']:
                    raise ValueError('Unexpected checkpoint size: ' + name)
                staged = stage / name
                with bundle.open(member) as incoming, staged.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if sha256(staged) != models[name]['sha256']:
                    raise ValueError('Checkpoint SHA-256 mismatch: ' + name)

        destination.mkdir(exist_ok=True)
        for name in missing:
            # Atomic publication without overwriting any existing file.
            os.link(stage / name, destination / name)
            print('Installed and verified:', destination / name)
    print('Original models ready. Use the recorded inference recipe in the README.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
