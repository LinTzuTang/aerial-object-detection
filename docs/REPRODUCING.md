# Reproducing the project

Run commands from the repository root. Code, configurations, metrics and four example figures are included; raw data, prediction CSVs and model binaries are excluded.

## 1. Environment

Use Linux, Python 3.12 and one NVIDIA CUDA GPU supporting native BF16. Recorded runs used an L4 with PyTorch 2.11.0+cu130; the wheel requires a compatible driver.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -r requirements-viz.txt
```

Core packages include Ultralytics 8.4.173 and pycocotools 2.0.11. Use the root requirements file; vendored DEIM requirements are a historical upstream snapshot. DEIM sources and WBF are bundled with their notices. Allow at least 20 GB beyond the dataset for the environment, recovery states and predictions. The code enforces an observed 11,400 MiB GPU memory limit.

## 2. Dataset

Obtain the original assignment dataset separately, preserving the split and class IDs 0–9:

```text
/DATA/
├── sample_submission.csv
├── classes.txt
├── images/{train,val,test}/
├── labels/{train,val}/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

There are 4,500 train, 700 validation and 1,800 test images. Only train updates parameters. Data preparation checks annotation hashes, writes a YOLO YAML and converts DEIM image IDs to 1-based annotation order. It does not change original images, labels or category IDs.

### Optional: use a separately obtained original model archive

```bash
bash download.sh /path/to/hw1_r12946009_models_public035010.zip
```

The [checkpoint guide](../checkpoints/README.md) describes the original 317.5 MB archive and hash-checked installation into `checkpoints/`. This portfolio does not publish a model download link. If you already have the original archive, install it, skip Section 3, and use `--recipe configs/final_ensemble.json` in Section 4. Section 3 is for training new weights in a fresh copy.

## 3. Retrain all five checkpoints

From a fresh checkout with no training outputs or trained checkpoint files:

```bash
bash src/retrain_all.sh /DATA
```

This prepares data, downloads and verifies official generic pretrained models, runs all stages below, then registers five new weights in `checkpoints/` and writes `outputs/retrained_deployment.json`. It does not need an old Colab VM.

For stage-by-stage execution:

```bash
python src/prepare_dataset.py --data-root /DATA --output outputs/data
python src/download_pretrained.py --output outputs/pretrained

python src/train.py --data outputs/data/dataset.yaml --stage initial \
  --weights outputs/pretrained/yolo26l.pt --project outputs/train --name initial --device 0

python src/train.py --data outputs/data/dataset.yaml --stage gentle \
  --weights outputs/train/initial/weights/best.pt --project outputs/train --name gentle --device 0

python src/train.py --data outputs/data/dataset.yaml --stage fresh \
  --weights outputs/pretrained/yolo26l.pt --project outputs/train --name fresh1280 \
  --stop-after 66 --device 0

python src/train.py --data outputs/data/dataset.yaml --stage mosaic_control \
  --weights outputs/train/fresh1280/epoch_037_ema.pt \
  --project outputs/train --name mosaic03 --device 0

python src/train_deim.py --data-root /DATA --prepared outputs/data \
  --pretrained outputs/pretrained/deim_model.safetensors \
  --output outputs/train/deim --mode recorded

python src/train.py --data outputs/data/dataset.yaml --stage preprocess_clahe \
  --weights outputs/train/gentle/weights/best.pt \
  --project outputs/train --name clahe5 --device 0

python src/register_retrained.py \
  --gentle outputs/train/gentle/weights/best.pt \
  --fresh outputs/train/fresh1280/epoch_037_ema.pt \
  --mosaic outputs/train/mosaic03/epoch_010_ema.pt \
  --deim outputs/train/deim/best_ema.pth \
  --clahe outputs/train/clahe5/terminal.pt \
  --output outputs/retrained_deployment.json
```

| Stage | Initialization | Retained output |
|---|---|---|
| Initial → Gentle | Official YOLO26l → Initial best | Gentle validation-best EMA |
| Fresh | Official YOLO26l, independent optimizer | Explicit epoch-37 EMA |
| Mosaic 0.3 | Fresh37, new optimizer | Fixed epoch-10 EMA |
| DEIMv2-L | Official DEIMv2-L COCO with semantic head mapping | Native-valid-selected EMA |
| CLAHE | Gentle, new optimizer, fixed luminance transform | Fixed epoch-5 terminal EMA |

Fresh retains its 100-epoch cosine schedule and stops at epoch 66; shortening the schedule to 66 changes the experiment. CLAHE is required for the final ensemble: LAB L channel, clip limit 2, grid 8×8, blend 0.5, before resizing at training and inference. Its optional matched control uses `--stage preprocess_original` with a new output name.

DEIM `--mode recorded` reconstructs 10 whole epochs plus retained prefixes of 2,048 and 1,024 minibatches, totaling 3,204 updates. The original experiment stopped for budget before early stopping. `--mode early-stop` is a separate experiment (minimum 6 epochs, patience 2, min_delta 0.003, maximum 15). The submitted configuration retains epoch 10; the later epoch-11 comparison is validation-only. [Detailed training protocol](TRAINING_PROTOCOL.md) describes learning-rate transitions, initialization and update accounting.

### Resume interrupted training

Keep the full run directory, including recovery states and its best model:

```bash
python src/train.py --data outputs/data/dataset.yaml --stage fresh \
  --project outputs/train --name fresh1280 --stop-after 66 --device 0 \
  --resume outputs/train/fresh1280/recovery_last.pt

python src/train_deim.py --data-root /DATA --prepared outputs/data \
  --output outputs/train/deim --mode recorded \
  --resume outputs/train/deim/recovery_last.pth
```

YOLO resumes at epoch boundaries; DEIM resumes after the latest completed validation segment. Incomplete segments are repeated. Original Colab loader/RNG interruption order cannot be replayed exactly. The portable workflow has not been rerun end to end on a GPU in this documentation revision. Retraining does not promise identical weights or historical scores; evaluate new weights separately.

## 4. Generate submission_test.csv

After retraining and registration:

```bash
python src/infer.py \
  --images /DATA/images/test \
  --sample /DATA/sample_submission.csv \
  --recipe outputs/retrained_deployment.json \
  --workdir outputs/members \
  --output outputs/submission_test.csv \
  --device 0
```

For original trained checkpoints, use `download.sh` as described above and select `--recipe configs/final_ensemble.json`, which checks their original hashes. Generic pretrained models cannot directly replace these trained checkpoints.

The entry point loads models sequentially and generates six pools:

1. Gentle1536 + horizontal flip.
2. Gentle1280 + horizontal flip.
3. Mosaic epoch10 at 1536 + horizontal flip.
4. Fresh37 at 1280 + horizontal flip.
5. DEIM epoch10 at native 1280, no flip.
6. CLAHE epoch5 with CLAHE input at 1536 + horizontal flip.

Each pool uses class-wise NMS IoU 0.70. WBF uses IoU 0.75, six equal weights of 0.5 and `box_and_model_avg`. FP32, confidence 0.001, score/box decimals 6/3 and a global cap of 300 are preserved. This entry point requires the recorded sample hash and 1,800 unique test images.

Work directory and output CSV must be new paths. Columns are `image_id,PredictionString`, ordered by the sample CSV. Each box is `class_id confidence x_min y_min width height` in original-image pixels; empty predictions use `none`.

Historical final CSV: submission 57199616, public AP 0.35010, SHA-256 `ec64915b8b8932cbed235491d4a07302d2c842a40807cc936474031ac9e7764e`. It was selected from five validation-frozen candidates. The portfolio does not distribute its full prediction CSV. Hardware and numerical differences can affect regenerated predictions.

## 5. Evaluate saved validation predictions

Run canonical evaluation separately from the DEIM engine:

```bash
python src/evaluate.py \
  --annotation /DATA/annotations/instances_val.json \
  --predictions outputs/valid_predictions.json \
  --output outputs/valid_metrics.json
```

Input is a list of COCO detections with numeric `image_id`, `category_id`, `score` and pixel-space `bbox: [x,y,width,height]`. Annotation IDs map to `1..700` in annotation order; prediction IDs must match. This evaluator does not accept a Kaggle CSV directly. Prediction generation is separate.

Metrics use pycocotools 2.0.11, IoU 0.50:0.05:0.95 and `maxDets=[1,10,100]` after the prediction pipeline's 300-box cap. Calibration and audit subsets were reused; neither is an independent final test.

## 6. Inspect data and visualize predictions

```bash
python src/analyze_dataset.py --data-dir /DATA --output-dir outputs/eda

python src/visualize.py predictions \
  --images /DATA/images/test \
  --predictions outputs/submission_test.csv \
  --image-ids img_1ae8b7b991128bc7 img_2e9b763b691129bf \
  --confidence 0.25 --output-dir outputs/visualizations

python src/visualize.py clahe \
  --image /DATA/images/val/img_8c313c2da1b4dad3.jpg \
  --output outputs/clahe_comparison.png
```

These CPU tools produce dataset plots, annotated samples, saved-prediction figures, and an original/CLAHE comparison. See [Data processing and visualization](VISUALIZATION.md) for output formats, validation ground-truth comparisons, and a Python preprocessing example. Use fresh output paths; the drawing commands preserve source images and predictions.
