# Data processing and visualization

Run these commands from the repository root, replacing `/DATA` with the original assignment dataset. The data preparation, EDA, and drawing tools run on CPU. They read source files and write to separate output paths.

The full environment is described in [REPRODUCING.md](REPRODUCING.md). For EDA, also install:

```bash
python -m pip install -r requirements-viz.txt
```

## 1. Prepare training formats

```bash
python src/prepare_dataset.py \
  --data-root /DATA \
  --output outputs/data_preview
```

The dataset must contain `images/{train,val,test}`, `labels/{train,val}`, `classes.txt`, `sample_submission.csv`, and `annotations/instances_{train,val}.json`.

The [preparation script](../src/prepare_dataset.py) checks the original annotation hashes and split membership, then writes:

| Output | Purpose |
|---|---|
| `dataset.yaml` | YOLO paths and the ten class names |
| `train_numeric.json`, `val_numeric.json` | COCO annotations with integer image IDs for DEIM |
| `train_id_mapping.json`, `val_id_mapping.json` | Original image ID to numeric ID mapping |
| `prepared.json` | Preparation metadata |

Image IDs become `1..N` in the original annotation's image order. Category IDs stay `0..9`. No image resize or annotation geometry change happens here. Training reads the train split; validation remains for evaluation and selection.

`outputs/data_preview` is optional. The full `retrain_all.sh` workflow creates its own `outputs/data`; do not prepare that same directory before running the complete workflow.

## 2. Inspect class balance, box sizes, and labels

```bash
python src/analyze_dataset.py \
  --data-dir /DATA \
  --output-dir outputs/eda \
  --seed 3407 \
  --samples-per-split 4
```

The [EDA script](../src/analyze_dataset.py) checks train/validation image and label IDs, normalized YOLO fields, and agreement between YOLO and COCO per-class annotation totals. For test, it compares image count with the sample CSV row count.

| Output | Contents |
|---|---|
| `summary.json` | Image/annotation counts, class counts, image dimensions, box density, area bands, occlusion, and truncation |
| `class_distribution.png` | Train/validation annotation counts by class |
| `density_and_box_scale.png` | Boxes per image and square root of box area in original-image pixels |
| `samples_train.jpg`, `samples_val.jpg` | Seeded ground-truth examples with boxes and class names |

The sample grids resize images for display and scale boxes accordingly. They show supplied annotations. Model predictions are drawn separately in the next section. The original-pixel area chart does not measure the fraction of boxes smaller than eight pixels after model resizing.

## 3. Draw saved model predictions

First generate `outputs/submission_test.csv` using the [inference instructions](REPRODUCING.md#4-generate-submission_testcsv). Then draw selected images:

```bash
python src/visualize.py predictions \
  --images /DATA/images/test \
  --predictions outputs/submission_test.csv \
  --image-ids img_1ae8b7b991128bc7 img_2e9b763b691129bf \
  --confidence 0.25 \
  --output-dir outputs/visualizations
```

Use filename stems without `.jpg` for `--image-ids`. Add `--boxes-only` to hide text labels in dense scenes. Every prediction at or above the display threshold is drawn, subject to clipping at image boundaries; no additional detection-count cap is applied by the drawing tool.

Each requested image produces a JPEG. `provenance.json` records image/prediction hashes, display thresholds, counts, and output hashes. The tool loads no model and leaves the source CSV unchanged. Retrained predictions may differ from the historical figures in `assets/`.

### Ground truth beside validation predictions

If you have saved validation detections in COCO JSON format:

```bash
python src/visualize.py predictions \
  --images /DATA/images/val \
  --predictions outputs/valid_predictions.json \
  --annotation /DATA/annotations/instances_val.json \
  --id-space numeric \
  --image-ids img_4d40a3f55e8ddd87 img_8c313c2da1b4dad3 \
  --confidence 0.25 \
  --output-dir outputs/valid_comparisons
```

Ground truth appears on the left and predictions on the right, at the same image scale. Validation prediction generation is a separate step; this command only reads an existing file. Plain `.json` and `.json.gz` are accepted.

| Format | Image identification | Bounding boxes |
|---|---|---|
| Kaggle CSV | `image_id` is the filename stem | `PredictionString`: repeated `class_id confidence x y width height`, or `none` |
| COCO JSON, `--id-space numeric` | `image_id` is `1..N` in original annotation order | `bbox: [x, y, width, height]`, with `category_id` and `score` |
| COCO JSON, `--id-space original` | `image_id` matches the original annotation ID | Same COCO detection fields |

All boxes use original-image pixel coordinates and classes `0..9`. COCO JSON requires `--annotation` to map IDs to filenames. CSV can also be combined with `--annotation` for a ground-truth panel; `--id-space` applies only to JSON predictions.

The display threshold of **0.25** is for readable examples. Evaluation/submission uses the fixed prediction pipeline with **0.001** confidence and its **300-box cap**. Changing the drawing threshold does not recalculate AP or change submitted predictions.

## 4. Preview the actual CLAHE transform

```bash
python src/visualize.py clahe \
  --image /DATA/images/val/img_8c313c2da1b4dad3.jpg \
  --output outputs/clahe_comparison.png
```

This produces a full-resolution original/processed PNG and a JSON file recording the policy and file hashes. It calls the same [luminance preprocessing](../src/luminance_clahe.py) used in the CLAHE model branch: LAB L channel, clip limit 2, an 8×8 tile grid, and a 50/50 blend with original luminance, before resizing.

To reuse that transform in Python from the repository root:

```python
import numpy as np
from PIL import Image
from src.luminance_clahe import preprocess

with Image.open("/DATA/images/val/img_8c313c2da1b4dad3.jpg") as image:
    rgb = np.asarray(image.convert("RGB"))
enhanced_rgb = preprocess(rgb, color_order="RGB")
enhanced_image = Image.fromarray(enhanced_rgb)
```

Declare `color_order="BGR"` for images loaded through OpenCV. The transform changes luminance while preserving image geometry, so box coordinates stay the same. Training/inference already apply it in the CLAHE branch; use original source images to avoid applying it twice.

All examples use fresh output paths. This documentation update adds portable EDA/drawing entry points; those entry points have not yet been executed as part of this update.
