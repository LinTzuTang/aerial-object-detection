# Archived DEIM training workers

These files preserve the original Colab experiment sources:

- `deimv2_l_remote.py`: model configuration, annotation conversion, training utilities, and the initial pilot.
- `deimv2_revision_remote.py`: semantic COCO-to-target head initialization, layer-specific optimizer groups, and recovery snapshots.
- `deimv2_earlystop_remote.py`: continuation from optimizer/EMA state, canonical validation, and budget/early-stop handling.

The original workers expect `/content/cvpdl_postprocess`, a pinned DEIM source checkout, pretrained safetensors, dataset annotations, runtime budget JSON, and recovery checkpoints. Some files also enforce the recorded L4 environment. Those runtime artifacts are not included here.

Helper imports `postprocess_policy` and `luminance_clahe` are retained in `src/`. When adapting the workers, include that directory in the Python import path and adapt the original filesystem assumptions. Refer to [the recorded recipe](../../configs/deim_training_recipe.json) for training parameters; it is a summary, not a complete worker launch plan.

The final inference pipeline is the separate, packaged entry point `src/infer.py`.

For new training, use portable `src/train_deim.py` or `src/retrain_all.sh` as described in [the reproduction guide](../../docs/REPRODUCING.md). The portable entry point reuses initialization/grouping helpers and manages data paths and recovery locally.
