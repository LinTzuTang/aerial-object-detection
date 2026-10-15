# Experiment record

[Project overview](../README.md) · [Recorded tables in JSON](../results/recorded_tables.json)

These tables preserve the configurations and scores recorded in the completed assignment report. All AP values use a 0–1 scale. The public results are Kaggle feedback; the validation results use the 700-image labeled validation split.

## Evaluation and selection

- Only 4,500 training images update model parameters.
- Validation is used for checkpoint and inference selection. A 350-image calibration subset selects fusion/view settings; the reused 350-image audit subset and full validation set provide confirmation, not independent holdout estimates.
- Each final candidate recipe was fixed on validation before its test fusion. The last five submissions use cached predictions; public feedback chooses the final artifact. The public winner does not have the highest validation AP.
- Canonical validation uses pycocotools 2.0.11 with IoU 0.50:0.05:0.95 and maxDets 1/10/100. AP-small uses the COCO small-object area range. Prediction pipelines retain at most 300 boxes unless a historical row specifies another cap.
- Runs use a single seed. Small changes are descriptive observations, not significance claims.

## Selected conclusions

| Comparison | Observation | Decision |
|---|---|---|
| Gentle 1280 vs. 1536 + flip | Valid AP +0.01594; AP-small +0.02533 | Retain larger inference and scale diversity |
| Four YOLO sets vs. adding DEIM epoch 10 | Valid AP about +0.00842; public AP +0.00908 | Retain the mixed-detector ensemble |
| Mosaic 0.3 vs. mosaic 0 continuation | Each arm has 889 successful optimizer updates; mosaic 0.3 gives the stronger final ensemble | Use mosaic 0.3 epoch 10 |
| CLAHE vs. original-image continuation on unchanged validation | Matched 538 successful updates per arm; AP −0.00246; AP-small −0.00494 | Negative standalone result; evaluate ensemble complementarity separately |
| Add CLAHE epoch5 with CLAHE input as a sixth pool | Valid AP +0.00144; AP-small +0.00369; public AP +0.00147 | Retain equal six-pool WBF, IoU 0.75 |
| DEIM epoch 11 in the same frozen fusion | Valid AP about +0.00012; AP-small about −0.00007 | Keep the submitted epoch-10 recipe |

The small-object analysis found 26.44% of training boxes below eight pixels on the short side at 960, versus 15.53% at 1280. The Fresh experiment also changes other training settings, so it does not isolate resolution as the cause of its result.

The low-light analysis uses 63 validation images with mean luminance below 70 as a brightness proxy, not verified night labels. The original pipeline scores 0.34829 AP on this subset; the CLAHE-trained/CLAHE-input pipeline scores 0.33468.

DEIM initialization maps matching COCO classification rows to target classes; the new van/tricycle/awning-tricycle rows remain randomly initialized. Classification/detection/backbone learning rates are 0.0005/0.0001/0.00001. Initialization and learning rates change together, so this is not an isolated ablation of either change.

DEIM continuation finishes 11 whole epochs, with 3,486 actual optimizer updates including interrupted prefixes. It stops because the budget cannot fit another whole epoch. Early stopping (minimum 6 epochs, patience 2, min_delta 0.003) is not reached. The submitted ensemble uses epoch 10; epoch 11 is validation-only.

## Training settings for the YOLO branches

| Parameter | Gentle branch | Fresh branch |
| --- | --- | --- |
| Initialization | Initial HW1 run: official weights; continuation: initial best | Official yolo26l.pt; independent optimizer |
| Resolution / physical batch / nbs | 960 / 4 / 64 | 1280 / 2 / 64 |
| Optimizer / initial LR | Initial: auto -> AdamW / 0.000714; continuation: AdamW / 0.0001 | AdamW / 0.0003 |
| Schedule / weight decay | Cosine / 0.0005 | Cosine / 0.0005 |
| Epochs / selected checkpoint | Initial: 30, best 22; continuation: 9, best 1 | 100 planned; stopped at 66; selected epoch 37 |
| Warmup / patience | Initial: 3 / 8; continuation: 1 / 8 | 5 / 0; stopped manually after stagnation |
| Mosaic / scale / translate | Initial: 1.0 / 0.5 / 0.1; continuation: 0 / 0.2 / 0.1 | 0.3 / 0.2 / 0.05; close_mosaic=15 |
| HSV h/s/v | 0.015 / 0.7 / 0.4 | 0.015 / 0.3 / 0.3 |
| Shared settings | Seed 3407; flip 0.5; MixUp/CutMix/Copy-Paste 0; native-resolution continuation loader | Same seed, flip and disabled mixing; no class-frequency loss weights |

## All completed public submissions

Training and inference sizes are recorded separately. The last five rows are the October-15 cached-prediction fusion study.

| Method | Train px | Inference px | Public AP |
|---|---|---|---|
| YOLO11m | 960 | 960 | 0.26594 |
| YOLO11m | 960 | 1280 | 0.29138 |
| YOLO11m; confidence 0.01 | 960 | 1280 | 0.29023 |
| YOLO11m; confidence 0.05 | 960 | 1280 | 0.28612 |
| YOLO11m; confidence 0.10 | 960 | 1280 | 0.28158 |
| YOLO26m | 960 | 1280 | 0.29044 |
| D-FINE-M | 960 | 960 | 0.25063 |
| YOLO26l initial | 960 | 1280 | 0.29918 |
| YOLO26l initial | 960 | 960 | 0.27320 |
| RT-DETR R50vd; Objects365 -> COCO | 640 | 640 | 0.24639 |
| YOLO26l initial; max 500 detections | 960 | 1280 | 0.29834 |
| YOLO26l initial; overlapping tiles / max 500 | 960 | Tiles -> 960 | 0.29365 |
| YOLO26l gentle continuation | 960 | 1280 | 0.30215 |
| YOLO26l mixed-crop fine-tune | 960 | 1280 | 0.30127 |
| YOLO26l high-resolution fine-tune (10 epochs) | 1280 | 1280 | 0.29271 |
| RF-DETR-L 2026 | 704 | 704 | 0.23232 |
| YOLO26l high-resolution + small crops (15 epochs) | 1280 | 1280 | 0.29626 |
| YOLO26l gentle; stricter bicycle NMS | 960 | 1280 | 0.30216 |
| YOLO26l gentle; single-model best | 960 | 1536 + flip | 0.31851 |
| YOLO26l fresh initialization; epoch 37 | 1280 | 1536 + flip | 0.30650 |
| YOLO26l P2 head | 960 | 1536 + flip | 0.27881 |
| Gentle + Fresh; two-view WBF IoU 0.70 | 960 / 1280 | 1536 + flip | 0.33266 |
| Three-view WBF IoU 0.70; add Gentle1280 | 960 / 1280 | 1280/1536 + flip | 0.33574 |
| Four-view WBF IoU 0.75; previous submission | 960 / 1280 | 1280/1536 + flip | 0.33877 |
| Four-view WBF; mosaic 0.3 epoch 10 replaces Fresh1536 | 960 / 1280 | 1280/1536 + flip | 0.33955 |
| YOLO26l original-image control; fixed epoch5 | 1280 | 1536 + flip | 0.31736 |
| YOLO26l CLAHE5; CLAHE input | 1280 | 1536 + flip | 0.31639 |
| CLAHE YOLO; 1280/1536 flip WBF | 1280 | 1280/1536 + flip | 0.32336 |
| Four-view WBF; CLAHE replaces mosaic member | 960 / 1280 | 1280/1536 + flip | 0.33906 |
| Five-view WBF; add DEIMv2-L | 960 / 1280 | 1280/1536 + flip | 0.34863 |
| Six-pool WBF; CLAHE w0.25, IoU0.70 | 960 / 1280 | 1280/1536 + flip | 0.34968 |
| Six-pool WBF; original5 w0.25, IoU0.70 | 960 / 1280 | 1280/1536 + flip | 0.34946 |
| Six-pool WBF; original5 w0.5, IoU0.75 | 960 / 1280 | 1280/1536 + flip | 0.35006 |
| Six-pool WBF; CLAHE w0.125, IoU0.70 | 960 / 1280 | 1280/1536 + flip | 0.34926 |
| Final six-pool WBF; CLAHE w0.5, IoU0.75 | 960 / 1280 | 1280/1536 + flip | 0.35010 |

## Recorded full-validation configurations

These include historical tuning, matched continuations, single models, and ensembles; they are not all controlled one-variable comparisons.

| Method | Inference | Valid AP | AP-small |
|---|---|---|---|
| Gentle; max300 | 1280 | 0.33757 | 0.26512 |
| Gentle; larger inference + flip | 1536 + flip | 0.35351 | 0.29044 |
| 1280 fine-tune; 10 epochs | 1280 | 0.33574 | 0.25860 |
| 1280 + small crops; 15 epochs | 1280 | 0.33942 | 0.26364 |
| Gamma/exposure training | 1536 + flip | 0.35179 | 0.28203 |
| Class-frequency loss weighting | 1536 + flip | 0.35003 | 0.28061 |
| P2 head; best epoch 25 | 1536 + flip | 0.31129 | 0.25473 |
| Gamma0.8 inference confirmation | 1536 + flip | 0.35287 | 0.29043 |
| Two-model WBF | 1536 + flip | 0.37190 | 0.29838 |
| Three-view WBF; add Gentle1280 | 1280/1536 + flip | 0.37707 | 0.30134 |
| Previous four-view WBF | 1280/1536 + flip | 0.38066 | 0.30439 |
| Fresh37 baseline | 1536 + flip | 0.34290 | 0.26532 |
| Mosaic 0.3 continuation; fixed epoch 10 | 1536 + flip | 0.34391 | 0.27035 |
| Mosaic 0 continuation; fixed epoch 10 | 1536 + flip | 0.34347 | 0.26833 |
| Prior best: mosaic 0.3 replacement | 1280/1536 + flip | 0.38316 | 0.30679 |
| Four-view: mosaic 0 replacement | 1280/1536 + flip | 0.38305 | 0.30718 |
| Add clean1536 weight 0.25; not submitted | 1280/1536 + flip | 0.38375 | 0.30792 |
| DEIMv2-L original recipe; 1 epoch + NMS | 1280 | 0.03135 | 0.02053 |
| DEIM revised; selected epoch 10 | 1280 native | 0.31369 | 0.23145 |
| DEIM revised; same EMA + NMS0.70 | 1280 | 0.31400 | 0.23236 |
| DEIM extra; epoch 11; valid only | 1280 native | 0.31444 | 0.23200 |
| Extra DEIM in fixed five-view WBF; valid only | 1280/1536 + flip | 0.39170 | 0.31282 |
| Train original / valid original | 1536 + flip | 0.35806 | 0.28870 |
| Train original / valid CLAHE | 1536 + flip | 0.35456 | 0.28500 |
| Train CLAHE / valid original | 1280 + flip | 0.34654 | 0.26849 |
| Train CLAHE / valid CLAHE | 1280 + flip | 0.34813 | 0.27025 |
| Train CLAHE / valid original | 1536 + flip | 0.35560 | 0.28376 |
| Train CLAHE / valid CLAHE | 1536 + flip | 0.35677 | 0.28575 |
| CLAHE YOLO; 1280/1536 flip WBF | 1280/1536 + flip | 0.36726 | 0.29507 |
| Four-view WBF; CLAHE replaces mosaic member | 1280/1536 + flip | 0.38032 | 0.30707 |
| Five-view WBF; add DEIMv2-L | 1280/1536 + flip | 0.39158 | 0.31288 |
| Six-pool WBF; CLAHE w0.25, IoU0.70 | 1280/1536 + flip | 0.39351 | 0.31584 |
| Six-pool WBF; original5 w0.25, IoU0.70 | 1280/1536 + flip | 0.39365 | 0.31626 |
| Six-pool WBF; original5 w0.5, IoU0.75 | 1280/1536 + flip | 0.39320 | 0.31668 |
| Six-pool WBF; CLAHE w0.125, IoU0.70 | 1280/1536 + flip | 0.39306 | 0.31478 |
| Final six-pool WBF; CLAHE w0.5, IoU0.75 | 1280/1536 + flip | 0.39302 | 0.31657 |

## Final five submissions (October 15)

All rows append one pool to Gentle1536, Gentle1280, Mosaic1536, Fresh1280 and DEIM1280. Original5 is the matched original-image continuation; CLAHE5 uses CLAHE input. Base-pool weights remain 0.5.

| Extra pool | Weight | WBF IoU | Valid AP | AP-small | Public AP | Submission |
|---|---|---|---|---|---|---|
| CLAHE5 | 0.25 | 0.7 | 0.39351 | 0.31584 | 0.34968 | 57199588 |
| Original5 | 0.25 | 0.7 | 0.39365 | 0.31626 | 0.34946 | 57199593 |
| Original5 | 0.5 | 0.75 | 0.39320 | 0.31668 | 0.35006 | 57199600 |
| CLAHE5 | 0.125 | 0.7 | 0.39306 | 0.31478 | 0.34926 | 57199609 |
| CLAHE5 | 0.5 | 0.75 | 0.39302 | 0.31657 | 0.35010 | 57199616 |

The selected public result is **0.35010**, submission **57199616**, using CLAHE5 weight 0.5 / WBF IoU 0.75. Its full-valid AP is 0.39302 and AP-small is 0.31657. The highest-valid candidate instead reaches 0.39365 and public 0.34946. No private leaderboard score or ranking is claimed.

## Calibration search (24 configurations)

The search ran on cached predictions and used zero Colab units. Sixteen initial configurations were followed by eight refinements after AP75 failures; the audit subset was reused to motivate the second round. Eight candidates received full confirmation, and five passed the gates for submission. This audit is not an independent holdout. No DEIM flip or new training was run.

Calibration/full AP gain must be at least 0.0005; audit AP must improve. AP-small may not drop more than 0.002, audit/full AP75 more than 0.001, or any audit/full class AP more than 0.01. The calibration-only pass column below does not imply full acceptance.

| Configuration | Cal AP | Cal AP75 | Cal AP-small | Cal gate |
|---|---|---|---|---|
| baseline | 0.40091 | 0.43745 | 0.32211 | False |
| deim_0p75 | 0.39965 | 0.43655 | 0.32005 | False |
| deim_1p0 | 0.39727 | 0.43370 | 0.31777 | False |
| deim_1p5 | 0.39272 | 0.42956 | 0.31303 | False |
| iou_0p7 | 0.40162 | 0.43765 | 0.32410 | True |
| iou_0p8 | 0.39751 | 0.43701 | 0.31792 | False |
| gentle_each_half | 0.39993 | 0.43477 | 0.32051 | False |
| omit_fresh | 0.40027 | 0.43471 | 0.32536 | False |
| omit_mosaic | 0.39600 | 0.42809 | 0.31901 | False |
| vehicles_0p5 | 0.40064 | 0.43620 | 0.32293 | False |
| vehicles_1p5 | 0.40181 | 0.43816 | 0.32449 | True |
| vehicles_2p0 | 0.40126 | 0.43779 | 0.32435 | False |
| original1536_original_w0p125 | 0.40234 | 0.43899 | 0.32502 | True |
| original1536_original_w0p25 | 0.40248 | 0.43934 | 0.32587 | True |
| clahe1536_clahe_w0p125 | 0.40202 | 0.43777 | 0.32498 | True |
| clahe1536_clahe_w0p25 | 0.40252 | 0.44047 | 0.32537 | True |
| p75_original1536_original_w0p125 | 0.40142 | 0.43900 | 0.32287 | True |
| p75_original1536_original_w0p25 | 0.40188 | 0.43953 | 0.32454 | True |
| p75_original1536_original_w0p5 | 0.40213 | 0.44054 | 0.32529 | True |
| p75_clahe1536_clahe_w0p125 | 0.40126 | 0.43820 | 0.32279 | False |
| p75_clahe1536_clahe_w0p25 | 0.40171 | 0.44076 | 0.32359 | True |
| p75_clahe1536_clahe_w0p5 | 0.40199 | 0.44129 | 0.32493 | True |
| p75_both_w0p125 | 0.40169 | 0.43969 | 0.32436 | True |
| p75_both_w0p25 | 0.40181 | 0.44138 | 0.32538 | True |

Increasing the global DEIM weight, halving Gentle's weights, or removing Fresh/Mosaic did not improve calibration AP. Lower WBF IoU and vehicle class reweighting could improve AP while failing the audit AP75 gate. Adding an original or CLAHE continuation offered useful complementary predictions; combining both as seven pools did not outperform the best six-pool calibration result. Full metrics and frozen specifications are in `results/final_fusion_calibration.json` and `results/final_fusion_cycle.json`.
