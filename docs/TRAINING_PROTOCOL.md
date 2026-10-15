# 重訓配方與原實驗的對應

## 執行入口

操作指令集中於根目錄 [REPRODUCING.md](REPRODUCING.md)。`retrain_all.sh` 依序建立所有必要 checkpoint；新增入口不依賴先前 Colab VM。本次整理沒有執行新的 GPU 訓練或 inference，因此沒有新的訓練驗證分數。

|階段|初始化|結束／選擇方式|供下一階段使用的輸出|
|---|---|---|---|
|Initial|官方 YOLO26l|30 epochs 上限，patience 8，valid best|`outputs/train/initial/weights/best.pt`|
|Gentle|Initial best|20 epochs 上限，patience 8，valid best|`outputs/train/gentle/weights/best.pt`|
|Fresh1280|官方 YOLO26l|100-epoch LR schedule，原探索至 66 epochs|`outputs/train/fresh1280/epoch_037_ema.pt`|
|Mosaic 0.3|Fresh37 EMA，新 AdamW|固定 10 epochs 的 EMA|`outputs/train/mosaic03/epoch_010_ema.pt`|
|DEIMv2-L|官方 DEIMv2-L COCO|native full-valid AP 選 EMA|`outputs/train/deim/best_ema.pth`|
|CLAHE5|Gentle best，新 AdamW|固定 5 epochs 的終點 EMA|`outputs/train/clahe5/terminal.pt`|

`train.py` 的 callback 以 1-based epoch 明確匯出 Fresh37 與 Mosaic10，避免 Ultralytics checkpoint 內部 epoch 索引或 `best.pt` 選模造成混淆。此整理新增的輸出與保存功能不改變各階段的訓練超參數。

## DEIM 的可攜重建

`train_deim.py` 重用原始 `deimv2_revision_remote.py` 的分類頭初始化與 optimizer 分組函式，模型／增強來源為 `records/deim_epoch10_effective_config.yml`。本地訓練 loop 掌控實際 LR、batch 與 EMA；YAML 中繼承自上游的 stock optimizer、scheduler、epoches 欄位不會驅動這個 loop。

### 初始化與更新

- 官方 repo commit：`1d2ca42171570c713e78fc6a766ec5104b7f4724`。
- 官方 HF model：`Intellindust/DEIMv2_DINOv3_L_COCO`，revision `6ce90ce18c14271c8ce145c35626d9a6544f881b`。
- 目標類別 → COCO 連續類別列：`0→0, 1→0, 2→1, 3→2, 5→7, 8→5, 9→3`；4、6、7 保留新初始化，denoising background 80→10。
- AdamW：分類頭 0.0005、其他偵測層 0.0001、DINOv3 backbone 0.00001，weight decay 0.000125；bias／norm 等參數不套 decay。
- 1280 正方形 resize、batch 2、accumulation 8、seed 3407、EMA 0.9998／warmups 500、gradient clip 0.1。
- RandomPhotometricDistort 0.3、RandomIoUCrop 0.3、水平翻轉 0.5；mosaic／mixup／copyblend 0；matching／augmentation switch 固定 10000。
- BF16 forward、FP32 loss 與 native valid inference。每段 valid 使用隔離程序的 pycocotools 2.0.11、maxDets `[1,10,100]`；類別 ID 與座標不重新映射。test 不參與訓練或選模。

### `--mode recorded` 的更新時序

|順序|圖片 pass|optimizer updates|LR 時程|
|---|---|---:|---|
|1|完整 epochs 1–2|564|0.25 epoch warmup；原 5-epoch cosine 的前 2 epochs|
|2|epoch 2 後額外 2,048 minibatches|256|以上一步最後 LR 為 base，13-pass cosine 前綴|
|3|完整 epochs 3–8|1,692|保留 AdamW／EMA，以前綴最後 LR 重設 base，13-pass cosine 從 0 到 6|
|4|epoch 8 後額外 1,024 minibatches|128|延續 cosine，位置 6 到 6＋1024/2250|
|5|完整 epochs 9–10|564|沿用同一 base，從 6＋1024/2250 接續|
|合計|25,572 minibatches／51,144 image samples|3,204|不使用 test 回饋調整|

每個完整 epoch 有 2,250 minibatches；最後不足 8 batches 的梯度組以實際大小平均，所以有 282 次 updates。兩個前綴分別為 256 與 128 次。前綴之後開始新的 shuffled full pass，符合原紀錄的額外曝光處理；不將前綴算成完成的 whole epoch。

舊 Colab 的多次 process 重啟、worker RNG 與 loader 中斷順序無法由目前輕量包精確還原。可攜入口統一 `workers=0`，重建資料曝光量與 LR 轉換，並不宣稱產生逐 bit 相同的權重。原報告中的 valid AP 0.39302／public 0.35010 對應原 checkpoint 與原提交 CSV（本作品集不附預測 CSV），不能直接當成本次重訓分數。

### 選模、early stop 與續跑

每個完整 pass／額外前綴後評估 native valid AP，保留 `best_ema.pth`。`recorded` 固定完成上述更新預算；`early-stop` 另以至少 6 whole epochs、patience 2、min_delta 0.003、最多 15 whole epochs 運作。原提交是訓練到 epoch 10 後因預算停止；之後的 epoch 11 沒有納入提交模型。

每段驗證完成後保存 `recovery_last.pth`，包含 raw model、EMA、AdamW、RNG、LR base、更新計數與下一段索引。保留同一目錄的 `best_ema.pth` 與紀錄再以 `--resume` 續跑。若當段尚未完成驗證便中斷，恢復時重做該段；不宣稱逐 minibatch 無損續跑。

YOLO 的 `recovery_last.pt` 使用 Ultralytics 原生 epoch 續跑語義，不是逐 minibatch replay；`--stage` 必須與原 run 一致。最終 CLAHE 分支及可選的 5-epoch 原圖 matched 對照應各從同一 Gentle 重新開始，途中失敗時重開該 arm，以保持相同更新計數。

## 推論 recipe 的雜湊

`configs/final_ensemble.json` 與 `results/summary.json` 是原提交的來源紀錄，含原 checkpoint SHA。輕量包不包含這些 checkpoint。

`register_retrained.py` 會將新權重放到 `checkpoints/` 並輸出新的 `outputs/retrained_deployment.json`。它只換權重 SHA，保留六個 prediction pools（包含 CLAHE 模型及同樣的 CLAHE 輸入）、解析度、flip、NMS 與 WBF 設定；原 CSV SHA 改列為參考，不宣稱新 CSV 與原 CSV 相同。推論仍逐一檢查新權重 SHA。

上游 DEIM 與 WBF 授權檔保留於 `vendor/`；資料集與自訓權重均未放入本輕量壓縮檔。
