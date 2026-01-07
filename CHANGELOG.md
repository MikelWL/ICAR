# Changelog

## v1.1.0 (2026-01-07)

- ICC: tune a val-calibrated routing threshold during `icc.train` and embed it into exported `ICC.pt` (`hparams.threshold`).
- ICAR eval: default to the embedded ICC threshold when `--icc-threshold` is not provided (falls back to 0.5 for older checkpoints).
- Scripts/docs: add `scripts/remap_icc_annotations_to_local.py` and document LAION ICC label remapping workflow.
- Config: set `model.icc_threshold: null` in `icar/configs/coco.yaml` to prefer checkpoint threshold.

## v1.0.0

- Initial public release of ICAR code, evaluation scripts, ICC training/export code, and IC9600 benchmark reproduction.
- Weights: `https://huggingface.co/MikelWL/icar-weights`, `https://huggingface.co/MikelWL/icc-weights`
- Data artifacts: `https://huggingface.co/datasets/MikelWL/icar-data`
