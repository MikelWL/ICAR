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

## Data

ICAR uses the standard Karpathy splits for COCO and Flickr30k. Download the
images from the official sources and the Karpathy split annotations from:

- Karpathy splits: https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

### COCO Retrieval Setup

1) Download COCO 2014 images:
   - train2014: http://images.cocodataset.org/zips/train2014.zip
   - val2014: http://images.cocodataset.org/zips/val2014.zip
2) Download and extract `caption_datasets.zip` from the Karpathy link above.
3) Organize the dataset as:

```
/path/to/coco-images/
  train2014/
    COCO_train2014_000000000009.jpg
    ...
  val2014/
    COCO_val2014_000000000042.jpg
    ...
  dataset_coco.json
```

### Flickr30k Retrieval Setup

1) Request and download Flickr30k images:
   - https://forms.illinois.edu/sec/229675
2) Download and extract `caption_datasets.zip` from the Karpathy link above.
3) Organize the dataset as:

```
/path/to/flickr30k/
  flickr30k-images/
    2923475135.jpg
    ...
  dataset_flickr30k.json
```

## Weights

Place checkpoints under:

```
checkpoints/
```

Weights are hosted on Hugging Face:

- ICAR weights: https://huggingface.co/MikelWL/icar-weights
- ICC weights: https://huggingface.co/MikelWL/icc-weights

Checksums are provided in each HF repo (`SHA256SUMS`).

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

## Licensing and Upstream Models

- Code in this repo is licensed under Apache 2.0 (see `LICENSE`).
- Weights are released separately and follow their upstream licenses:
  - ICAR weights (OpenCLIP ViT-L-14) inherit the OpenCLIP MIT license.
  - ICC weights (ConvNeXt V2) are CC-BY-NC due to ImageNet-pretrained weights.

Upstream repositories:
- OpenCLIP: https://github.com/mlfoundations/open_clip
- ConvNeXt V2: https://github.com/facebookresearch/ConvNeXt-V2

## Citation

```bibtex
@article{williams2025image,
  title={Image Complexity-Aware Adaptive Retrieval for Efficient Vision-Language Models},
  author={Williams-Lekuona, Mikel and Cosma, Georgina},
  journal={arXiv preprint arXiv:2512.15372},
  year={2025}
}
```

If you use the pretrained backbones, please also cite:

```bibtex
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

```bibtex
@article{Woo2023ConvNeXtV2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Sanghyun Woo and Shoubhik Debnath and Ronghang Hu and Xinlei Chen and Zhuang Liu and In So Kweon and Saining Xie},
  year={2023},
  journal={arXiv preprint arXiv:2301.00808},
}
```
