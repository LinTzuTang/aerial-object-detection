#!/usr/bin/env bash
# Run from the unpacked aerial-object-detection directory after installing requirements.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: bash src/retrain_all.sh /absolute/path/to/dataset" >&2
  exit 2
fi
data_root="$1"
if [ ! -f src/train.py ]; then
  echo "Run this command from the unpacked aerial-object-detection directory" >&2
  exit 2
fi
python src/prepare_dataset.py --data-root "$data_root" --output outputs/data
python src/download_pretrained.py --output outputs/pretrained
python src/train.py --data outputs/data/dataset.yaml --stage initial --weights outputs/pretrained/yolo26l.pt --project outputs/train --name initial --device 0
python src/train.py --data outputs/data/dataset.yaml --stage gentle --weights outputs/train/initial/weights/best.pt --project outputs/train --name gentle --device 0
python src/train.py --data outputs/data/dataset.yaml --stage fresh --weights outputs/pretrained/yolo26l.pt --project outputs/train --name fresh1280 --stop-after 66 --device 0
python src/train.py --data outputs/data/dataset.yaml --stage mosaic_control --weights outputs/train/fresh1280/epoch_037_ema.pt --project outputs/train --name mosaic03 --device 0
python src/train_deim.py --data-root "$data_root" --prepared outputs/data --pretrained outputs/pretrained/deim_model.safetensors --output outputs/train/deim --mode recorded
python src/train.py --data outputs/data/dataset.yaml --stage preprocess_clahe --weights outputs/train/gentle/weights/best.pt --project outputs/train --name clahe5 --device 0
python src/register_retrained.py \
  --gentle outputs/train/gentle/weights/best.pt \
  --fresh outputs/train/fresh1280/epoch_037_ema.pt \
  --mosaic outputs/train/mosaic03/epoch_010_ema.pt \
  --deim outputs/train/deim/best_ema.pth \
  --clahe outputs/train/clahe5/terminal.pt \
  --output outputs/retrained_deployment.json
echo "Training outputs registered. Follow docs/REPRODUCING.md for test inference."
