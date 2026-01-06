#!/usr/bin/env python3
"""
Materialize a local ICAR-compatible LAION subset after downloading with img2dataset.

Why this exists:
- We publish a manifest (url + original_id + caption + filename) for a fixed 100k
  LAION-COCO subset.
- A user can download from that manifest via img2dataset, but some URLs will be
  missing (erosion) and filenames/layout can vary.
- This script builds a contiguous local subset with filenames:
    images/laion_00000000.jpg ... images/laion_XXXXXXXX.jpg
  and regenerates a Karpathy-style JSON for the actually-downloaded images.

Expected input image naming:
- This script looks for downloaded images by LAION `original_id` (zero-padded),
  e.g. 000127314.jpg, anywhere under --img-root (recursive).

Outputs:
- <out_root>/images/*.jpg
- <out_root>/dataset_laion_coco_100k.json    (Karpathy format)
- <out_root>/url_to_local.json               (canonical mapping)
- <out_root>/original_id_to_local.json       (secondary mapping)
- Optional: remapped ICC annotations aligned to local filenames
"""

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ManifestRow:
    filename: str
    original_id: str
    url: str
    caption: str


def _iter_manifest_rows(path: Path) -> Iterable[ManifestRow]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield ManifestRow(
                filename=row["filename"],
                original_id=str(row["original_id"]),
                url=row.get("url", ""),
                caption=row.get("caption", ""),
            )


def _index_downloaded_images(img_root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Build lookups from:
    - LAION original_id -> absolute image path
    - URL -> absolute image path

    This supports two common layouts:
    1) Files named by original_id (e.g. 000127314.jpg)
    2) img2dataset shard layout (e.g. 00000/00000.jpg) where original_id is stored
       in the per-sample metadata JSON (requires --save_additional_columns).
    """
    by_original_id: Dict[str, Path] = {}
    by_url: Dict[str, Path] = {}
    # First, check for an explicit original_id-named layout.
    for p in img_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        stem = p.stem
        if stem.isdigit() and stem not in by_original_id:
            by_original_id[stem] = p

    # Then, index img2dataset metadata JSON files (authoritative when present).
    for meta_path in img_root.rglob("*.json"):
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("status") != "success":
            continue
        url = meta.get("url")
        original_id = meta.get("original_id")
        # Image is stored next to the metadata file with the same key/stem.
        img_path = meta_path.with_suffix(".jpg")
        if not img_path.exists():
            img_path = meta_path.with_suffix(".jpeg")
        if not img_path.exists():
            img_path = meta_path.with_suffix(".png")
        if not img_path.exists():
            img_path = meta_path.with_suffix(".webp")
        if img_path.exists():
            if original_id is not None:
                by_original_id[str(original_id)] = img_path
            if url:
                by_url[str(url)] = img_path
    return by_original_id, by_url


def _tokens_from_caption(caption: str) -> List[str]:
    return caption.lower().split()


def _make_karpathy_entry(
    idx: int,
    filename: str,
    caption: str,
    laion_metadata: Optional[Dict[str, str]] = None,
) -> Dict:
    sentence = {
        "tokens": _tokens_from_caption(caption),
        "raw": caption,
        "imgid": idx,
        "sentid": idx,
    }
    entry = {
        "imgid": idx,
        "filename": filename,
        "split": "test",
        "sentids": [idx],
        "sentences": [sentence],
    }
    if laion_metadata:
        entry["laion_metadata"] = laion_metadata
    return entry


def _load_icc_annotations(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="TSV manifest path")
    parser.add_argument(
        "--img-root",
        type=Path,
        required=True,
        help="Root directory containing img2dataset-downloaded images (searched recursively)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output directory to create (images + regenerated JSON)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of successfully materialized images",
    )
    parser.add_argument(
        "--icc-annotations",
        type=Path,
        default=None,
        help="Optional ICC annotations keyed by original_id.jpg (e.g. data/annotations_icc_in_laion_coco_100k.json)",
    )
    parser.add_argument(
        "--icc-annotations-by-url",
        type=Path,
        default=None,
        help="Optional ICC annotations keyed by url (e.g. data/laion_coco_100k/annotations_icc_in_laion_coco_100k_by_url.json)",
    )
    args = parser.parse_args()

    downloaded_by_id, downloaded_by_url = _index_downloaded_images(args.img_root)

    images_dir = args.out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    karpathy_images: List[Dict] = []
    id_to_local: Dict[str, Dict[str, str]] = {}
    url_to_local: Dict[str, Dict[str, str]] = {}

    icc_annotations: Optional[Dict[str, int]] = None
    if args.icc_annotations:
        icc_annotations = _load_icc_annotations(args.icc_annotations)
    icc_annotations_by_url: Optional[Dict[str, int]] = None
    if args.icc_annotations_by_url:
        icc_annotations_by_url = _load_icc_annotations(args.icc_annotations_by_url)
    icc_local: Dict[str, int] = {}

    kept = 0
    for row in _iter_manifest_rows(args.manifest):
        if args.max_images is not None and kept >= args.max_images:
            break

        original_id = row.original_id
        if not original_id:
            continue

        src = downloaded_by_id.get(original_id) or downloaded_by_url.get(row.url)
        if src is None:
            continue

        local_idx = kept
        local_filename = f"laion_{local_idx:08d}.jpg"
        dst = images_dir / local_filename

        shutil.copyfile(src, dst)

        laion_metadata = {"url": row.url, "original_id": original_id}
        karpathy_images.append(_make_karpathy_entry(local_idx, local_filename, row.caption, laion_metadata))
        id_to_local[original_id] = {"filename": local_filename, "imgid": str(local_idx)}
        if row.url:
            url_to_local[row.url] = {"filename": local_filename, "imgid": str(local_idx)}

        if icc_annotations_by_url is not None and row.url in icc_annotations_by_url:
            icc_local[local_filename] = int(icc_annotations_by_url[row.url])
        elif icc_annotations is not None:
            key = f"{original_id}.jpg"
            if key in icc_annotations:
                icc_local[local_filename] = int(icc_annotations[key])

        kept += 1

    dataset_json = {"dataset": "laion_coco_100k", "images": karpathy_images}
    (args.out_root / "dataset_laion_coco_100k.json").write_text(json.dumps(dataset_json, indent=2))
    (args.out_root / "original_id_to_local.json").write_text(json.dumps(id_to_local, indent=2, sort_keys=True))
    (args.out_root / "url_to_local.json").write_text(json.dumps(url_to_local, indent=2, sort_keys=True))

    if icc_annotations is not None or icc_annotations_by_url is not None:
        (args.out_root / "annotations_icc_local.json").write_text(json.dumps(icc_local, indent=2, sort_keys=True))

    print(f"Materialized {kept} images into {args.out_root}")


if __name__ == "__main__":
    main()
