# Trained checkpoints

The original project checkpoints are not distributed with this portfolio, and no model download URL is configured. Use the [full retraining workflow](../docs/REPRODUCING.md) to obtain new models. The hashes below identify the original models used for the recorded scores.

If you have separately obtained the matching original model ZIP, install it from the repository root:

```bash
bash download.sh /path/to/hw1_r12946009_models_public035010.zip
```

The installer verifies the ZIP and individual checkpoint hashes, then installs the models here. It reuses correct existing files and refuses to overwrite differing ones. It can also accept an authorized Drive file share URL using [gdown](https://github.com/wkentaro/gdown/tree/v5.2.0), installed by `requirements.txt`. Expected hashes are stored in [src/records/model_download.json](../src/records/model_download.json).

Archive: `hw1_r12946009_models_public035010.zip`, 317,535,211 bytes. SHA-256: `1b7b8769d1b8ab2509719f052a9494775e42708a251d3ae1fb6416a4eb1bc4d1`. Allow about 700 MiB for downloading and extraction.

| File | Role | SHA256 |
|---|---|---|
| `best.pt` | Gentle YOLO26l; both scales | `bbaf78e4e62269cb84f8a47381e93e0967e90721a9bab3e98efce7f49a482d49` |
| `mosaic03_epoch10.pt` | Mosaic 0.3 YOLO26l epoch 10 | `ff17fefc698242c85f413aa1671ae6e05cbdfc35bdd657ef3e78fef388d6e1fb` |
| `fresh_epoch37.pt` | Fresh YOLO26l epoch 37 | `24e09350d1529ffc777f36cd94f02e419988572aac129ddb20ec03c8eb4e93eb` |
| `deim_best_ema.pth` | DEIMv2-L epoch 10 EMA | `83d009c280e9ce25372b3f13a0f4206ba717b098ef8cbe5b2d351f57c54bdd4b` |
| `clahe_epoch5.pt` | CLAHE YOLO26l epoch 5 terminal EMA | `1bb231eaa93013f66522a95d8f95047b662f973ca73ac1c79c269bdd4e192985` |

Use the files from the original completed project artifacts. The entry point checks these hashes before inference. To train new weights, follow [the portable retraining workflow](../docs/REPRODUCING.md). `src/register_retrained.py` copies all five new checkpoints here and writes `outputs/retrained_deployment.json` with their new hashes. Use that recipe for retrained inference; these recorded hashes describe the original submitted models.
