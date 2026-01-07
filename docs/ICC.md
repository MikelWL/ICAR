## ICC (Image Complexity Classifier)

ICAR uses an **Image Complexity Classifier (ICC)** to decide whether an image is routed through an early-exit or full compute path during evaluation.

### Threshold selection (val-calibrated default)

When training ICC via `python -m icc.train`, the script will (by default) tune a probability threshold on the **val split** (metric configurable) and embed it into the exported `ICC.pt` under `hparams.threshold`. Evaluation scripts use this embedded threshold by default when `--icc-threshold` is not provided.

### Training ICC on LAION-COCO ICC labels (3021 subset)

The released ICC labels are a **subset** of the LAION-COCO 100k distractors (3021 URL-keyed labels; includes some `-1` rejected samples which ICC training drops by default).

Canonical workflow:
- Run `scripts/materialize_laion_subset_from_img2dataset.py` with `--icc-annotations-by-url ...`; it will produce `annotations_icc_local.json` aligned to the materialized `images/` folder.

If you already have a dataset JSON with `images[*].laion_metadata.url` and local `images[*].filename`, you can remap URL-keyed labels to filename-keyed labels with:
- `python scripts/remap_icc_annotations_to_local.py --annotations-by-url data/laion_coco_100k/annotations_icc_in_laion_coco_100k_by_url.json --dataset-json /path/to/dataset_laion_coco_100k.json --out /path/to/annotations_icc_local.json`

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
