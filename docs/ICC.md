## ICC (Image Complexity Classifier)

ICAR uses an **Image Complexity Classifier (ICC)** to decide whether an image is routed through an early-exit or full compute path during evaluation.

### Using the released `ICC.pt`

If you just want to reproduce ICAR results, download `ICC.pt` from:
- `https://huggingface.co/MikelWL/icc-weights`

ICAR expects an inference checkpoint in a simple PyTorch format (see `icar/models/icc.py`):
- `ICC.pt` is a dict with:
  - `state_dict` containing only `backbone.*` and `classifier.*`
  - optional `hparams` (e.g. `model_name`, `num_classes`, `feature_dim`)

### Training ICC from scratch (Lightning)

The ICC training implementation lives under `icc/` and will train, evaluate, and export an ICAR-compatible `ICC.pt`.

**Dependencies (training only)**
- `pytorch-lightning`, `torchmetrics`, `scikit-learn` (in addition to the main repo requirements)

Example:
```bash
pip install pytorch-lightning torchmetrics scikit-learn
```

**Dataset format**
- A folder of images (e.g. `images/`)
- An annotations JSON mapping `filename -> label` where `label ∈ {0, 1}` (and optionally `-1` for rejected samples)

Example:
```json
{
  "laion_00001234.jpg": 1,
  "laion_00005678.jpg": 0
}
```

**Train + test + export**
```bash
python -m icc.train \
  --data_dir /path/to/images \
  --annotations_file /path/to/annotations.json \
  --output_dir runs/icc_run \
  --model_name convnextv2_tiny.fcmae_ft_in22k_in1k \
  --img_size 224 \
  --batch_size 32 \
  --num_epochs 10 \
  --learning_rate 1e-5
```

Outputs:
- `runs/icc_run/checkpoints/final_tested.ckpt` (Lightning checkpoint)
- `runs/icc_run/ICC.pt` (ICAR-compatible inference checkpoint)
- `runs/icc_run/split_indices.json` (deterministic split indices for reproducibility)

**Export only (`.ckpt` → `ICC.pt`)**
```bash
python -m icc.export \
  --ckpt /path/to/final_tested.ckpt \
  --out ICC.pt
```

