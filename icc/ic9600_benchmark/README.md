## IC9600 Benchmark Reproduction (Handoff)

This folder enables reproducing the **IC9600 benchmark** (ConvNeXt-V2 regression variant trained/evaluated on IC9600, comparable to ICNet).

### Python dependencies (minimum)

- `torch`, `torchvision`
- `timm`
- `pytorch-lightning`
- `torchmetrics`
- `numpy`
- `scikit-learn`
- `scipy` (for Pearson/Spearman; required)
- `pandas` (CSV outputs)

### Required data files (IC9600 dataset)

You need a local IC9600 dataset directory with:

- `train.txt` (IC9600 split file; lines are `image_name␠␠score`)
- `test.txt`  (IC9600 split file; lines are `image_name␠␠score`)
- `images/`   (directory containing image files referenced by the txt files)

Example layout:

```
IC9600/
  train.txt
  test.txt
  images/
    000001.jpg
    000002.jpg
    ...
```

Notes:
- Scores are expected to already be normalized to `[0, 1]` (as in the canonical IC9600 release).
- The dataset txt files use a **double-space** separator (IC9600 convention).

### Train (ConvNeXt regression)

Produces a Lightning checkpoint and logs in `--output_dir`.

```bash
python -m icc.ic9600_benchmark.train \
  --ic9600_train_txt /path/to/IC9600/train.txt \
  --ic9600_test_txt  /path/to/IC9600/test.txt \
  --ic9600_img_dir   /path/to/IC9600/images \
  --output_dir runs/ic9600_convnext_regression \
  --model_name convnextv2_nano.fcmae_ft_in22k_in1k \
  --img_size 512 \
  --batch_size 32 \
  --num_epochs 30 \
  --learning_rate 1e-4
```

Outputs:
- `runs/.../checkpoints/best.ckpt` (best on `val_pearson`)
- `runs/.../checkpoints/last.ckpt`
- `runs/.../checkpoints/final_model.ckpt`
- `runs/.../run_config.json`

### Evaluate (ICNet-compatible metrics)

```bash
python -m icc.ic9600_benchmark.evaluate \
  --checkpoint runs/ic9600_convnext_regression/checkpoints/best.ckpt \
  --ic9600_test_txt /path/to/IC9600/test.txt \
  --ic9600_img_dir  /path/to/IC9600/images \
  --output_dir runs/ic9600_convnext_regression/eval_best
```

Outputs:
- `metrics.json` (RMSE, RMAE, Pearson, Spearman)
- `predictions.csv` (filename, ground_truth, prediction, error, abs_error)
