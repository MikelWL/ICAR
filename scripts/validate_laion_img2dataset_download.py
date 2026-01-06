#!/usr/bin/env python3
"""
Validate that a LAION download can be joined to the ICAR 100k subset manifest.

This script is intended as a pre-flight check for users who download the LAION
subset using img2dataset and want to materialize an ICAR-compatible folder
layout (with expected URL erosion).

It supports two common cases:
1) img2dataset output_format=files:
   - images stored as <shard>/<key>.jpg
   - metadata stored as <shard>/<key>.json containing at least: url, status
2) img2dataset output_format=webdataset:
   - shard bundles stored as <shard>.tar (or .tar.gz/.tgz), containing per-sample
     <key>.json and <key>.<ext> members
2) already-materialized ICAR layout:
   - images stored as images/laion_XXXXXXXX.jpg

The canonical join key used here is URL.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple


def _load_manifest_urls(manifest_path: Path) -> Tuple[Dict[str, Dict[str, str]], Set[str]]:
    by_url: Dict[str, Dict[str, str]] = {}
    duplicates: Set[str] = set()
    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"url", "filename", "original_id", "caption"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            url = str(row["url"]).strip()
            if not url:
                continue
            if url in by_url:
                duplicates.add(url)
            by_url[url] = {
                "filename": str(row["filename"]).strip(),
                "original_id": str(row["original_id"]).strip(),
                "caption": str(row["caption"]).strip(),
            }
    return by_url, duplicates


def _iter_json_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".json"):
                continue
            yield Path(dirpath) / name


def _associated_image_path(meta_path: Path) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = meta_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def _iter_tar_files(root: Path) -> Iterable[Path]:
    for p in root.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz"):
            yield p


def _scan_webdataset_tar(
    tar_path: Path,
    manifest_urls: Set[str],
    matched_urls: Set[str],
    *,
    progress_every: Optional[int] = None,
) -> Tuple[int, int, int]:
    scanned_json = 0
    meta_like = 0
    success_in_manifest = 0
    last_log = time.time()
    start = last_log

    mode = "r"
    lower = tar_path.name.lower()
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        mode = "r:gz"

    with tarfile.open(tar_path, mode) as tf:
        for member in tf:
            if not member.isfile():
                continue
            if not member.name.endswith(".json"):
                continue
            scanned_json += 1
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                meta = json.loads(f.read().decode("utf-8", errors="ignore"))
            except Exception:
                continue
            if not isinstance(meta, dict):
                continue
            if "url" not in meta:
                continue
            meta_like += 1
            url = str(meta.get("url") or "").strip()
            status = str(meta.get("status") or "").strip()
            if status and status != "success":
                continue
            if url in manifest_urls:
                success_in_manifest += 1
                matched_urls.add(url)

            if progress_every and scanned_json % progress_every == 0:
                now = time.time()
                dt = max(1e-6, now - last_log)
                elapsed = now - start
                print(
                    f"[{tar_path.name}] scanned_json={scanned_json} meta_like={meta_like} "
                    f"success_in_manifest={success_in_manifest} unique_urls={len(matched_urls)} "
                    f"(+{progress_every} in {dt:.1f}s, elapsed {elapsed/60:.1f}m)"
                )
                last_log = now

    return scanned_json, meta_like, success_in_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-root",
        type=Path,
        required=True,
        help="Root folder of img2dataset output (or an already-materialized ICAR LAION folder)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/laion_coco_100k/laion_coco_100k_manifest.tsv"),
        help="Path to the published LAION subset manifest TSV",
    )
    parser.add_argument(
        "--icc-annotations-by-url",
        type=Path,
        default=None,
        help="Optional ICC annotations keyed by URL (JSON mapping url->label)",
    )
    parser.add_argument(
        "--max-json",
        type=int,
        default=None,
        help="Optional cap on number of metadata JSON files to scan",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Log progress every N metadata items (applies to both sidecar JSON and webdataset tar JSON)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if join coverage is unexpectedly low",
    )
    args = parser.parse_args()

    manifest_by_url, duplicates = _load_manifest_urls(args.manifest)
    manifest_urls = set(manifest_by_url.keys())

    print(f"Manifest: {len(manifest_urls)} URLs (duplicates: {len(duplicates)})")

    # If the user already has an ICAR-style directory, validate count quickly.
    icar_images_dir = args.download_root / "images"
    if icar_images_dir.exists():
        laion_images = list(icar_images_dir.glob("laion_*.jpg"))
        if laion_images:
            print(f"Detected ICAR-style images/: {len(laion_images)} files")
            return 0

    # Otherwise, scan img2dataset metadata JSON files and join by URL.
    json_paths = _iter_json_files(args.download_root)
    scanned = 0
    meta_like = 0
    success = 0
    success_in_manifest = 0
    success_in_manifest_with_image = 0
    matched_urls: Set[str] = set()

    for meta_path in json_paths:
        scanned += 1
        if args.max_json is not None and scanned > args.max_json:
            break
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        if "url" not in meta or "status" not in meta:
            continue
        meta_like += 1
        url = str(meta.get("url") or "").strip()
        status = str(meta.get("status") or "").strip()
        if status != "success":
            continue
        success += 1

        if url in manifest_urls:
            success_in_manifest += 1
            matched_urls.add(url)
            if _associated_image_path(meta_path) is not None:
                success_in_manifest_with_image += 1

        if args.progress_every and scanned % args.progress_every == 0:
            print(
                f"Scanned {scanned} json | meta_like={meta_like} success={success} "
                f"success_in_manifest={success_in_manifest}"
            )

    if meta_like == 0:
        # Try output_format=webdataset (tar bundles)
        tar_files = list(_iter_tar_files(args.download_root))
        if tar_files:
            scanned_json = 0
            tar_meta_like = 0
            tar_success_in_manifest = 0
            for tar_path in tar_files:
                print(f"Scanning tar shard: {tar_path.name}")
                sj, ml, sim = _scan_webdataset_tar(
                    tar_path,
                    manifest_urls,
                    matched_urls,
                    progress_every=args.progress_every,
                )
                scanned_json += sj
                tar_meta_like += ml
                tar_success_in_manifest += sim
            print(f"Detected webdataset shards: {len(tar_files)} tar files")
            print(f"Scanned tar JSON: {scanned_json} (meta_like: {tar_meta_like})")
            print(f"Join coverage: success_in_manifest={tar_success_in_manifest} (unique_urls={len(matched_urls)})")
            if args.icc_annotations_by_url:
                ann = json.loads(args.icc_annotations_by_url.read_text())
                ann_urls = set(ann.keys())
                print(f"ICC labels (URL-keyed): {len(ann_urls)}")
                print(f"ICC labels ∩ manifest: {len(ann_urls & manifest_urls)}")
                print(f"ICC labels ∩ downloaded_success: {len(ann_urls & matched_urls)}")

            if args.strict:
                if len(matched_urls) < max(10, int(0.01 * len(manifest_urls))):
                    print("ERROR: join coverage is very low; check that you used the correct manifest.")
                    return 1
            return 0

        print("No img2dataset-style metadata JSON files found under --download-root.")
        print("Expected either:")
        print("- output_format=files: per-sample *.json containing {url, status}, or")
        print("- output_format=webdataset: shard *.tar containing per-sample *.json with {url} (and optionally status).")
        return 2

    print(f"Scanned JSON: {scanned} (meta_like: {meta_like})")
    print(f"Downloads: success={success}")
    print(f"Join coverage: success_in_manifest={success_in_manifest} (unique_urls={len(matched_urls)})")
    print(f"Images present next to JSON: {success_in_manifest_with_image}/{success_in_manifest}")

    if args.icc_annotations_by_url:
        ann = json.loads(args.icc_annotations_by_url.read_text())
        ann_urls = set(ann.keys())
        print(f"ICC labels (URL-keyed): {len(ann_urls)}")
        print(f"ICC labels ∩ manifest: {len(ann_urls & manifest_urls)}")
        print(f"ICC labels ∩ downloaded_success: {len(ann_urls & matched_urls)}")

    if args.strict:
        # Very conservative heuristic: if we matched less than 1% of the manifest,
        # something is likely misconfigured.
        if len(matched_urls) < max(10, int(0.01 * len(manifest_urls))):
            print("ERROR: join coverage is very low; check input_format/url_col and output_format.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
