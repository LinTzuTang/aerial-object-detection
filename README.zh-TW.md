# 航拍場景物件偵測

### 小物件、低亮度場景與多模型融合

[English](README.md) · [完整實驗結果](docs/EXPERIMENTS.md) · [執行說明](docs/REPRODUCING.md) · [資料與視覺化說明](docs/VISUALIZATION.md)

這個個人電腦視覺專案以航拍影像中的人與車輛為偵測目標，處理小物件、密集場景、光照差異與類別不平衡。工作涵蓋資料分析、模型微調、前處理對照實驗、驗證集調參與預測融合。

最終方法結合 **YOLO26l 與 DEIMv2-L**，以 **5 個 checkpoint 產生 6 組預測**再做加權框融合。Kaggle public AP 從最佳已提交單一模型的 **0.31851** 提升至 **0.35010**，絕對提升 **0.03159 AP**。

| 最終 Valid AP | Valid AP-small | Kaggle public AP |
|---:|---:|---:|
| **0.39302** | **0.31657** | **0.35010** |

AP 指 IoU 0.50:0.05:0.95 的 mAP；valid 與 Kaggle public 使用不同影像集合。

## 偵測成果

點擊圖片可查看完整尺寸的預測框與類別標籤。

| 日間市區 | 低亮度路口 |
|---|---|
| [![航拍市區的人與車輛偵測](assets/urban_traffic.jpg)](assets/urban_traffic.jpg) | [![低亮度路口的車輛偵測](assets/low_light_traffic.jpg)](assets/low_light_traffic.jpg) |
| 橋梁上的小型車輛 | 夜間密集場景 |
| [![橋梁航拍影像的小型車輛偵測](assets/bridge_traffic.jpg)](assets/bridge_traffic.jpg) | [![夜間密集場景的人與車輛偵測](assets/crowded_night.jpg)](assets/crowded_night.jpg) |

展示門檻為 confidence ≥ 0.25；評估與提交使用 0.001，最多保留 300 個框。上排為 test 範例，下排為 valid 範例。圖片用來呈現模型行為，不能將每個顯示的框都視為正確預測。[圖片來源紀錄](assets/provenance.json)

## 題目與資料

本專案源自 **臺大 Computer Vision Practice with Deep Learning 2026** 個人作業。輸入 RGB 影像，輸出每個物件的類別、信心分數與 bounding box。

| 資料切分 | 張數 | 用途 |
|---|---:|---|
| Train | 4,500 | 更新模型參數 |
| Valid | 700 | 選 checkpoint、調整推論參數與評估 |
| Test | 1,800 | 最終推論與 Kaggle 評分 |

十個類別為 `pedestrian`、`people`、`bicycle`、`car`、`van`、`truck`、`tricycle`、`awning-tricycle`、`bus`、`motor`。微調只使用作業訓練集，初始化使用通用資料集的預訓練模型。

## 我完成的工作

- 分析訓練框尺寸與亮度分布，據此規劃小物件及低亮度實驗。
- 建立獨立訓練及續訓的 YOLO26l 配方，調整 DEIMv2-L 分類頭初始化與分層學習率。
- 用相同更新預算比較原圖／CLAHE 訓練，以及不同 mosaic 設定。
- 實作多尺度、水平翻轉、class-wise NMS 與 Weighted Boxes Fusion 推論流程。
- 保存實驗分數、固定部署設定、權重雜湊與 CSV 格式，支援結果追溯及重現。

模型架構與通用預訓練權重來自上游專案；我的工作著重在任務適配、實驗設計、評估與整合。

## 最終模型與方法

| Checkpoint | 訓練方式 | 推論設定 |
|---|---|---|
| Gentle YOLO26l | 960 訓練後進行較溫和的續訓 | 1280＋flip、1536＋flip |
| Fresh YOLO26l epoch 37 | 獨立的 1280 訓練 | 1280＋flip |
| Mosaic 0.3 YOLO26l epoch 10 | 從 Fresh37 續訓 10 epochs | 1536＋flip |
| DEIMv2-L epoch 10 | COCO 類別對應初始化與分層學習率 | 原圖 1280＋NMS |
| CLAHE YOLO26l epoch 5 | 從 Gentle 以 CLAHE 圖片續訓 5 epochs | CLAHE 輸入，1536＋flip |

每個 YOLO 預測組包含未翻轉與水平翻轉兩個視角；CLAHE 分支在兩個視角前套用固定前處理，其餘分支使用原圖，先將座標映射回原圖，再做 class-wise NMS。DEIM 不使用 flip。Gentle 使用兩個尺度，因此總共是 **5 個 checkpoint、6 組預測**。

融合使用 WBF IoU **0.75**，六組權重皆為 **0.5**，信心分數計算方式為 `box_and_model_avg`。各組預測先將 score／座標保留 6／3 位小數，再做融合。[完整部署設定](configs/final_ensemble.json)

### CLAHE：調整局部對比

**CLAHE（Contrast Limited Adaptive Histogram Equalization）**會將影像分成局部區域，分別調整亮度分布，並限制對比增強程度，降低雜訊被過度放大的情況。[OpenCV 說明](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)

本專案先將 RGB／BGR 轉成 LAB，只處理 **L 亮度通道**，再混合 **50% 原始 L＋50% 增強 L**。固定使用 `clipLimit=2.0`、`tileGridSize=(8, 8)`，也就是將圖片水平與垂直各分成 8 格。在原始解析度、resize 之前處理，影像尺寸與框座標維持一致；CLAHE 分支在訓練與推論時使用相同設定。[前處理程式](src/luminance_clahe.py)

### WBF：融合重疊的預測框

**WBF（Weighted Boxes Fusion，加權框融合）**將同類別且高度重疊的框分組，以信心分數與各組預測權重對座標加權平均，讓多個預測共同決定最後的位置。[原始論文](https://arxiv.org/abs/1910.13302)

本專案先在每組預測內做 IoU **0.70** 的 class-wise NMS，再用 IoU **0.75** 做跨組 WBF。IoU 是兩框交集面積除以聯集面積，0.75 是判斷是否融合的重疊門檻。六組權重皆為 **0.5**，其中 Gentle 因兩個尺度而占兩組；融合後的信心分數使用 `box_and_model_avg`，考慮實際參與的預測組及其權重。[融合程式](src/fuse.py)

## 實驗成果

| 方法 | Valid AP | AP-small | Public AP |
|---|---:|---:|---:|
| Gentle，1280 | 0.33757 | 0.26512 | 0.30215 |
| Gentle，1536＋flip | 0.35351 | 0.29044 | 0.31851 |
| Gentle＋Fresh，兩組 WBF | 0.37190 | 0.29838 | 0.33266 |
| 加入 Gentle1280，三組 WBF | 0.37707 | 0.30134 | 0.33574 |
| 四組 WBF | 0.38066 | 0.30439 | 0.33877 |
| 四組 WBF，加入 mosaic 續訓模型 | 0.38316 | 0.30679 | 0.33955 |
| 五組 WBF，加入 DEIMv2-L | 0.39158 | 0.31288 | 0.34863 |
| **最終六組 WBF，加入 CLAHE YOLO** | **0.39302** | **0.31657** | **0.35010** |

加入 DEIM 後，valid AP 約提升 **0.00842**，public AP 提升 **0.00908**。DEIM 單獨的 valid AP 較低，但其預測能補充 YOLO，提升融合結果。

再加入 CLAHE 訓練與 CLAHE 輸入的預測組，valid AP 提升 **0.00144**、AP-small 提升 **0.00369**、public AP 提升 **0.00147**。最終依 public 選出的候選並不是 valid AP 最高的候選。

[完整紀錄](docs/EXPERIMENTS.md) 收錄 35 筆 public 提交、36 種完整 valid 設定，以及最後 24 組 calibration 搜尋，也包含沒有改善的實驗。

## 重要策略與發現

### 小物件需要保留更多影像細節

訓練框短邊小於 8 pixels 的比例，在 960 下為 **26.44%**，1280 下減少至 **15.53%**。Gentle 使用較大推論尺度與 flip 後，AP-small 從 **0.26512** 提升至 **0.29044**。另外也測試裁切、切圖與 P2 head，但沒有取代最終設定。Fresh 同時修改多項訓練條件，因此不能將結果全部歸因於解析度。

### 不同模型與尺度能提供互補預測

我使用 valid 選擇推論與融合設定，在各候選的 test 融合前固定配方。最後五次提交直接使用已保存的預測，不需新增 GPU 推論。最終融合同時包含尺度差異、不同訓練 checkpoint 與 transformer detector。權重是對每組預測設定，Gentle 以兩個尺度貢獻兩組。

### 圖片前處理需要用對照實驗判斷

原圖與 CLAHE 兩組從相同權重開始，使用相同 seed、5 epochs 及各 **538 次成功 optimizer 更新**。CLAHE 只處理 LAB 的 L 通道，clip limit 為 2、網格為 8×8、混合比例為 0.5。

在未前處理的 valid 上，CLAHE 訓練相對原圖訓練的 AP **下降 0.00246**，AP-small **下降 0.00494**，但在最後的融合實驗中，加入「CLAHE 模型＋CLAHE 輸入」能改善 ensemble，因此保留為第六組預測。單獨表現與加入融合後的互補效果是不同的比較。較早測試的 gamma／曝光增強與類別頻率加權，也沒有改善選定的 Gentle baseline。

### 在記憶體限制下執行

透過循序載入模型、gradient accumulation 與 CUDA 記憶體監測控制資源。原圖／CLAHE 對照訓練的記錄峰值為 **8,522.9 MiB**；本輪新增 test 推論的峰值為 **2,940.9 MiB**。這些數值對應特定實驗，尚未進行所有設定的效能或延遲 benchmark。

## 執行方式

以下指令從作品集根目錄執行，將 `/DATA` 改成原始作業資料集的絕對路徑。資料、訓練權重與輸出檔均排除於 Git。

### 1. 安裝環境

完整訓練使用 **Linux、Python 3.12，以及支援原生 BF16 的 NVIDIA CUDA GPU**。原實驗使用 L4 與 CUDA 13.0 的 PyTorch wheel，需有相容的驅動程式。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -r requirements-viz.txt
```

### 2. 資料處理與分析

```text
/DATA/
├── classes.txt
├── sample_submission.csv
├── images/{train,val,test}/
├── labels/{train,val}/
└── annotations/{instances_train.json,instances_val.json}
```

資料集已包含 YOLO 標註與 COCO JSON。[資料準備程式](src/prepare_dataset.py) 會檢查原始標註、建立 YOLO dataset YAML，並將影像 ID 轉成 DEIM 使用的整數；類別仍為 **0–9**，來源檔案維持原樣。

```bash
# 選用：先將處理結果輸出到獨立的預覽目錄。
python src/prepare_dataset.py --data-root /DATA --output outputs/data_preview

# 類別分布、框大小／密度，以及 train／valid 標註範例。
python src/analyze_dataset.py --data-dir /DATA --output-dir outputs/eda
```

[EDA 程式](src/analyze_dataset.py) 會產生 `summary.json`、`class_distribution.png`、`density_and_box_scale.png`、`samples_train.jpg`、`samples_val.jpg`。Train 用於參數更新；valid 用於模型選擇與調參。

### 3. 重新訓練五個 checkpoint

作品集提供完整重訓流程，沒有公開原始專案權重。原始權重的雜湊與另行取得後的安裝方式見 [權重說明](checkpoints/README.md)。

```bash
bash src/retrain_all.sh /DATA
```

[完整流程](src/retrain_all.sh) 會準備 `outputs/data`、下載並核對官方通用預訓練權重，依序執行 **Initial → Gentle**、獨立的 **Fresh1280 → Mosaic0.3**、**DEIMv2-L**、**Gentle → CLAHE**，最後將五個權重放入 `checkpoints/`，產生 `outputs/retrained_deployment.json`。

執行前須沒有同名訓練輸出與既有訓練權重；前面的 `outputs/data_preview` 與正式使用的 `outputs/data` 分開。分階段指令、續跑與訓練預算詳見 [重現說明](docs/REPRODUCING.md) 和 [訓練流程](docs/TRAINING_PROTOCOL.md)。這份可攜式流程尚未重新在 GPU 完整跑過；重訓得到的新權重需另行評估，分數可能不同。

### 4. 產生提交檔

完成重訓並登錄新模型後執行：

```bash
python src/infer.py \
  --images /DATA/images/test \
  --sample /DATA/sample_submission.csv \
  --recipe outputs/retrained_deployment.json \
  --workdir outputs/members \
  --output outputs/submission_test.csv \
  --device 0
```

輸出為六組預測 WBF 融合後的 CSV，每張 test 圖片一列。每次使用新的輸出路徑。若已取得雜湊符合紀錄的原始權重，改用 `--recipe configs/final_ensemble.json`。

### 5. 繪製預測與前處理對照

以下讀取已產生的 CSV，繪製上方兩張 test 範例的預測：

```bash
python src/visualize.py predictions \
  --images /DATA/images/test \
  --predictions outputs/submission_test.csv \
  --image-ids img_1ae8b7b991128bc7 img_2e9b763b691129bf \
  --confidence 0.25 \
  --output-dir outputs/visualizations

python src/visualize.py clahe \
  --image /DATA/images/val/img_8c313c2da1b4dad3.jpg \
  --output outputs/clahe_comparison.png
```

[視覺化程式](src/visualize.py) 在 CPU 執行，讀取已存的預測，並記錄來源雜湊。也支援 COCO JSON，以及標註／預測並排比較。`--confidence 0.25` 僅影響顯示，不會更改提交檔或評分門檻。CLAHE 指令使用實際訓練／推論的前處理，產生原圖與處理後圖片的對照。[格式、Python 範例與其他指令](docs/VISUALIZATION.md)

### 程式索引

| 功能 | 程式 |
|---|---|
| 安裝另行取得的原始權重壓縮檔 | [download.sh](download.sh) · [download_models.py](src/download_models.py) |
| 資料準備／EDA | [prepare_dataset.py](src/prepare_dataset.py) · [analyze_dataset.py](src/analyze_dataset.py) |
| 完整重訓／個別訓練 | [retrain_all.sh](src/retrain_all.sh) · [train.py](src/train.py) · [train_deim.py](src/train_deim.py) |
| CLAHE 前處理 | [luminance_clahe.py](src/luminance_clahe.py) |
| 推論／WBF／驗證指標 | [infer.py](src/infer.py) · [fuse.py](src/fuse.py) · [evaluate.py](src/evaluate.py) |
| 預測框／CLAHE 對照圖 | [visualize.py](src/visualize.py) |

## 限制與後續方向

- Valid 及其 calibration／audit 子集已被重複使用，實驗採單一 seed，因此小幅增益不代表獨立驗證或統計顯著。
- Kaggle public 分數是公開榜回饋，不能推定 private 成績或其他航拍資料集的泛化表現。
- DEIM 續訓至 epoch 11 因預算停止，**尚未達 early stop**。最終提交仍使用 epoch 10；epoch 11 融合僅在 valid 增加約 0.00012 AP，未另行提交。
- 推論策略以準確度為主，目前沒有量測吞吐量與部署延遲。
- 後續可比較相同更新預算下的解析度、以多個 seeds 確認改善，並量測延遲及嘗試將 ensemble 蒸餾成單一模型。

## 使用的上游專案

[Ultralytics YOLO](https://docs.ultralytics.com/models/yolo26)、[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)、[Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)。資料與評估題目由課程提供。第三方原始碼與模型保留各自的授權條款，見 [THIRD_PARTY.md](THIRD_PARTY.md)。
