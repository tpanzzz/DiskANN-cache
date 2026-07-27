#!/usr/bin/env python3
"""Prepare deterministic all-base and sampled-base self-query ground truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_fbin_metadata(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.read(8)
    if len(header) != 8:
        raise ValueError(f"{path} is too short to contain an fbin header")
    rows, dimensions = struct.unpack("<II", header)
    expected_size = 8 + rows * dimensions * np.dtype("<f4").itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes; expected {expected_size}"
        )
    return rows, dimensions


def write_identity_truthset(path: Path, node_ids: np.ndarray) -> None:
    with path.open("wb") as file:
        file.write(struct.pack("<II", len(node_ids), 1))
        np.asarray(node_ids, dtype="<u4").tofile(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("--sample-fraction must be in (0, 1]")

    rows, dimensions = read_fbin_metadata(args.base_file)
    sample_size = max(1, math.ceil(rows * args.sample_fraction))
    rng = np.random.default_rng(args.seed)
    sample_ids = rng.choice(rows, size=sample_size, replace=False).astype(np.uint32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_groundtruth = args.output_dir / "base_all.identity_groundtruth.bin"
    sample_query = args.output_dir / f"base_sample_{args.sample_fraction:g}_seed{args.seed}.fbin"
    sample_groundtruth = (
        args.output_dir
        / f"base_sample_{args.sample_fraction:g}_seed{args.seed}.identity_groundtruth.bin"
    )
    sample_ids_csv = (
        args.output_dir / f"base_sample_{args.sample_fraction:g}_seed{args.seed}.ids.csv"
    )

    write_identity_truthset(all_groundtruth, np.arange(rows, dtype=np.uint32))
    base = np.memmap(
        args.base_file,
        mode="r",
        dtype="<f4",
        offset=8,
        shape=(rows, dimensions),
    )
    with sample_query.open("wb") as file:
        file.write(struct.pack("<II", sample_size, dimensions))
        np.asarray(base[sample_ids], dtype="<f4").tofile(file)
    write_identity_truthset(sample_groundtruth, sample_ids)

    with sample_ids_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_rank", "node_id"])
        writer.writerows(enumerate(sample_ids.tolist(), start=1))

    manifest = {
        "base_file": str(args.base_file.resolve()),
        "base_rows": rows,
        "dimensions": dimensions,
        "all_query_file": str(args.base_file.resolve()),
        "all_groundtruth_file": str(all_groundtruth.resolve()),
        "sample_fraction": args.sample_fraction,
        "sample_size": sample_size,
        "seed": args.seed,
        "sample_query_file": str(sample_query.resolve()),
        "sample_groundtruth_file": str(sample_groundtruth.resolve()),
        "sample_ids_file": str(sample_ids_csv.resolve()),
        "output_sha256": {
            all_groundtruth.name: sha256(all_groundtruth),
            sample_query.name: sha256(sample_query),
            sample_groundtruth.name: sha256(sample_groundtruth),
            sample_ids_csv.name: sha256(sample_ids_csv),
        },
    }
    (args.output_dir / "workload_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
