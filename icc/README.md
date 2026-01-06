## ICC (Image Complexity Classifier)

This directory contains a clean, reproducible implementation of the **ConvNeXt-V2 binary image complexity classifier** ("simple vs complex") plus a deterministic data split and an **export step** that produces an ICAR-compatible `ICC.pt` inference checkpoint.

### What this module provides

- **Inference model**: `ConvNeXtICC` (pure PyTorch `nn.Module`, no Lightning required at runtime)
- **Training**: PyTorch Lightning training script for fine-tuning on `annotations.json` + image folder
- **Export**: convert a Lightning `.ckpt` into an inference-friendly `ICC.pt` with:
  - `state_dict` containing only `backbone.*` and `classifier.*` keys
  - `hparams` including `model_name`, `num_classes`, `feature_dim`
  - `model_architecture` string

### Dataset format

This implementation matches the format used in this repo:

- `annotations.json`: dict of `filename -> label` where `label ∈ {0, 1}` (and optionally `-1` to reject)
- `data_dir`: directory containing the referenced images

### Train (Lightning)

Example (mirrors the defaults used in this repo):

```bash
python -m icc.train \
  --data_dir data/images \
  --annotations_file data/annotations.json \
  --output_dir runs/icc_run \
  --model_name convnextv2_tiny.fcmae_ft_in22k_in1k \
  --img_size 224 \
  --batch_size 32 \
  --num_epochs 10 \
  --learning_rate 1e-5
```

At the end, the script writes:
- `output_dir/checkpoints/final_tested.ckpt` (Lightning)
- `output_dir/ICC.pt` (ICAR-compatible inference checkpoint)

### Export (ckpt -> ICC.pt)

```bash
python -m icc.export \
  --ckpt path/to/final_tested.ckpt \
  --out ICC.pt
```

