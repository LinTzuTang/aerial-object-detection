# Third-party sources

This repository presents an individual course project. The detector architectures, generic pretrained weights, and WBF algorithm are attributed to their upstream authors.

## Ultralytics YOLO

- Source: https://github.com/ultralytics/ultralytics
- Documentation: https://docs.ultralytics.com/models/yolo26
- Recorded package version: `8.4.173`.
- Initialization: official `yolo26l.pt`, generic Objects365 v1 followed by COCO pretraining.
- Installed as a Python dependency. Its upstream license applies.

## DEIMv2

- Source: https://github.com/Intellindust-AI-Lab/DEIMv2
- Recorded commit: `1d2ca42171570c713e78fc6a766ec5104b7f4724`.
- Pretrained detector: https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO
- Recorded model revision: `6ce90ce18c14271c8ce145c35626d9a6544f881b`.
- Pretraining: DINOv3 LVD-1689M general web images, then COCO object detection.
- Vendored inference sources: [src/vendor/DEIMv2](src/vendor/DEIMv2).
- License: [original DEIMv2 license](src/vendor/DEIMv2/LICENSE.md), including its non-commercial terms and third-party exceptions.

The selected vendored files are copied byte for byte from the recorded upstream checkout. Project-specific changes live outside that directory: target-class initialization, optimizer groups, training configuration, image transformations, inference, NMS, and WBF. The training workers also adapt the torchvision transform hook names at runtime.

## Weighted Boxes Fusion

- Source: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
- Recorded implementation: `ensemble-boxes` version `1.0.9`.
- Vendored file: [src/vendor/ensemble_boxes_wbf.py](src/vendor/ensemble_boxes_wbf.py).
- License: [MIT license and original copyright](src/vendor/ENSEMBLE_BOXES_LICENSE).

## Dataset and figures

The images are examples from the NTU CVPDL 2026 HW1 dataset, annotated with this project's final predictions. The full dataset and hidden evaluation annotations are not included. Image IDs, splits, display thresholds, and file hashes are recorded in [assets/provenance.json](assets/provenance.json).

Third-party materials retain their own terms; this repository does not relicense them under a single blanket license.
