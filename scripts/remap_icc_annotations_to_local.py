#!/usr/bin/env python3
"""
Remap URL-keyed ICC annotations to local image filenames.

Use this when:
- you have a Karpathy-style dataset JSON that contains per-image `laion_metadata.url`
  and the local `filename` (e.g. `dataset_laion_coco_100k.json`), and
- your ICC labels are keyed by URL (e.g. `annotations_icc_in_laion_coco_100k_by_url.json`).

Note: ICC training (`icc.data.dataset.ICCDataset`) drops label `-1` by default.
This script keeps `-1` labels unless `--drop-rejected` is set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--annotations-by-url",
        type=Path,
        required=True,
        help="Path to URL-keyed annotations JSON (url -> label).",
    )
    p.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="Karpathy-style dataset JSON containing `images[*].filename` and `images[*].laion_metadata.url`.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for filename-keyed annotations JSON (filename -> label).",
    )
    p.add_argument(
        "--drop-rejected",
        action="store_true",
        help="Drop label -1 entries when writing the output JSON.",
    )
    args = p.parse_args()

    annotations_by_url = _load_json(args.annotations_by_url)
    dataset = _load_json(args.dataset_json)

    url_to_filename = {}
    for img in dataset.get("images", []):
        meta = img.get("laion_metadata") or {}
        url = meta.get("url")
        filename = img.get("filename")
        if url and filename:
            url_to_filename[str(url)] = str(filename)

    out = {}
    missing = 0
    for url, label in annotations_by_url.items():
        filename = url_to_filename.get(str(url))
        if filename is None:
            missing += 1
            continue
        label_int = int(label)
        if args.drop_rejected and label_int == -1:
            continue
        out[filename] = label_int

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))

    print(f"Wrote {args.out} ({len(out)} entries; missing_urls={missing})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

