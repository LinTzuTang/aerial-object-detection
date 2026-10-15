#!/usr/bin/env bash
# Usage: bash download.sh 'GOOGLE_DRIVE_FILE_SHARE_URL'
set -euo pipefail
model_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$model_script_dir/src/download_models.py" "$@"
