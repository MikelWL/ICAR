# ICAR: Image Complexity-Aware Retrieval

<p align="center">
  <a href="https://arxiv.org/abs/2512.15372">
    <img src="https://img.shields.io/badge/paper-arXiv-blue" alt="Paper">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  </a>
</p>

Official implementation of the ECIR 2026 paper: **Image Complexity-Aware Adaptive Retrieval for Efficient Vision-Language Models**.

ICAR implements adaptive routing for image-text retrieval using an image complexity classifier (ICC).
The main evaluation setting reported in the paper uses COCO/Flickr test sets with 100k LAION-COCO
distractors in the retrieval set.

## Release Status

- Paper accepted at ECIR 2026 (December 2025)
- Repository initialized (December 2025)
- Initial implementation released (January 2026)
- Final release (weights + data guidance) planned before ECIR 2026

## Release Checklist

- ICAR and ICC pretrained weights release
- ICC-specific implementation release
- Data download instructions

## Setup

Create a conda environment, install requirements, and run scripts from this repository:

```bash
pip install -r requirements.txt
```

Update dataset paths in:

- `icar/configs/coco.yaml`
- `icar/configs/flickr30k.yaml`

## Weights

Place checkpoints under:

```
checkpoints/
```

Planned: download links + checksums will be provided here.

## Reference Scripts

These scripts are the primary entrypoints for training and evaluation. Use them as-is, or open them
to see the underlying Python calls and adjust flags for custom variants.

```bash
# Training (single-GPU baseline + variants)
scripts/train_single_gpu.sh coco all
scripts/train_single_gpu.sh flickr all

# Mixed eval (COCO/Flickr + 100k LAION distractors, instance + category)
scripts/eval_mixed.sh all
```

## Project Structure

```
icar/
├── models/          # Model implementations
├── data/           # Data loading and preprocessing
├── training/       # Training components
├── evaluation/     # Evaluation utilities
└── configs/        # Configuration files

scripts/            # Main scripts for training and evaluation
```

## Citation

```bibtex
@article{williams2025image,
  title={Image Complexity-Aware Adaptive Retrieval for Efficient Vision-Language Models},
  author={Williams-Lekuona, Mikel and Cosma, Georgina},
  journal={arXiv preprint arXiv:2512.15372},
  year={2025}
}
```
