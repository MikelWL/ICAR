## Data Guide

ICAR’s main results are reported on:
- **MS-COCO / Flickr30k test splits** (Karpathy splits)
- plus **100k LAION-COCO distractor images** (downloaded from URLs; erosion expected)

This repo does not ship images. It ships scripts + expects you to download images from upstream sources, and to download ICAR’s metadata/labels from Hugging Face.

### 0) Download ICAR data artifacts

Download the ICAR data artifacts repo into your local `data/` directory:
- HF dataset repo: `https://huggingface.co/datasets/MikelWL/icar-data`

This provides:
- COCO/Flickr category-eval labels
- LAION 100k manifest + LAION category labels
- ICC URL-keyed annotations (for ICC experiments)

If you have the Hugging Face CLI installed, one convenient option is:
```
hf download MikelWL/icar-data --repo-type dataset --local-dir data
```

### 1) COCO / Flickr (Karpathy splits)

Download:
- Karpathy splits (JSON): `https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip`
  - Extract and place:
    - `dataset_coco.json`
    - `dataset_flickr30k.json`
  under `data/caption_datasets/`.

Then download images from the official sources:

**COCO 2014**
- train2014: `http://images.cocodataset.org/zips/train2014.zip`
- val2014: `http://images.cocodataset.org/zips/val2014.zip`

Expected layout:
```
/path/to/coco-images/
  train2014/
  val2014/
data/caption_datasets/dataset_coco.json
```

**Flickr30k**
- Request form: `https://forms.illinois.edu/sec/229675`

Expected layout:
```
/path/to/flickr30k/
  flickr30k-images/
data/caption_datasets/dataset_flickr30k.json
```

Update dataset paths in:
- `icar/configs/coco.yaml`
- `icar/configs/flickr30k.yaml`

### 2) LAION-COCO distractors (100k subset)

The ICAR mixed evaluation uses **100,000 LAION-COCO distractors**. We publish:
- `data/laion_coco_100k/laion_coco_100k_manifest.tsv` (img2dataset manifest)
- `data/laion_coco_100k/category_labels_laion_coco_100k.json` (category labels for distractors)

Download the images using `img2dataset` from the manifest. Some URLs will fail; that’s expected.

#### Validate (pre-flight)

After downloading, run:
```
python scripts/validate_laion_img2dataset_download.py \
  --download-root /path/to/your/img2dataset/output
```

#### Materialize into ICAR layout

Materialize a contiguous local folder layout (and regenerate a matching `dataset_laion_coco_100k.json`) with:
```
python scripts/materialize_laion_subset_from_img2dataset.py \
  --manifest data/laion_coco_100k/laion_coco_100k_manifest.tsv \
  --img-root /path/to/your/img2dataset/output \
  --out-root /path/to/laion_coco_100k_materialized \
  --icc-annotations-by-url data/laion_coco_100k/annotations_icc_in_laion_coco_100k_by_url.json
```

This creates:
```
/path/to/laion_coco_100k_materialized/
  images/laion_00000000.jpg
  images/laion_00000001.jpg
  ...
  dataset_laion_coco_100k.json
  url_to_local.json
  annotations_icc_local.json  (optional)
```

Point evaluation at the materialized folder via `LAION_ROOT` (see `scripts/eval_mixed.sh`).
