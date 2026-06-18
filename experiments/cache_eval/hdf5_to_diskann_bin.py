#!/usr/bin/env python3
"""Convert ANN benchmark HDF5 datasets to DiskANN binary files.

Expected common HDF5 keys are train/test/neighbors/distances, but the script
prints available keys and lets callers override names because public mirrors
do not always use identical naming.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def main() -> None:
    args = parse_args()
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "h5py is required. Install it in the experiment environment, for example: "
            "python -m pip install h5py numpy"
        ) from exc

    with h5py.File(args.input, "r") as h5:
        keys = list(h5.keys())
        print("HDF5 keys:", ", ".join(keys))
        if args.inspect:
            for key in keys:
                dataset = h5[key]
                shape = getattr(dataset, "shape", None)
                dtype = getattr(dataset, "dtype", None)
                print(f"{key}: shape={shape} dtype={dtype}")
            return

        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        base = read_dataset(h5, args.base_key, "base/train vectors").astype(np.float32)
        queries = read_dataset(h5, args.query_key, "query/test vectors").astype(np.float32)
        write_diskann_matrix(output_dir / args.base_output, base)
        write_diskann_matrix(output_dir / args.query_output, queries)

        if args.neighbors_key:
            neighbors = read_dataset(h5, args.neighbors_key, "ground-truth neighbors").astype(
                np.uint32
            )
            distances = None
            if args.distances_key and args.distances_key in h5:
                distances = read_dataset(h5, args.distances_key, "ground-truth distances").astype(
                    np.float32
                )
            write_truthset(output_dir / args.groundtruth_output, neighbors, distances)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HDF5 ANN benchmark data to DiskANN binary files"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--base-key", default="train")
    parser.add_argument("--query-key", default="test")
    parser.add_argument("--neighbors-key", default="neighbors")
    parser.add_argument("--distances-key", default="distances")
    parser.add_argument("--base-output", default="base.fbin")
    parser.add_argument("--query-output", default="query.fbin")
    parser.add_argument("--groundtruth-output", default="groundtruth.bin")
    return parser.parse_args()


def read_dataset(h5, key: str, label: str) -> np.ndarray:
    if key not in h5:
        raise SystemExit(f"Missing {label} key '{key}'. Available keys: {', '.join(h5.keys())}")
    data = np.asarray(h5[key])
    if data.ndim != 2:
        raise SystemExit(f"Dataset '{key}' must be 2-D, got shape {data.shape}")
    return np.ascontiguousarray(data)


def write_diskann_matrix(path: Path, matrix: np.ndarray) -> None:
    matrix = np.ascontiguousarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] > np.iinfo(np.uint32).max or matrix.shape[1] > np.iinfo(np.uint32).max:
        raise ValueError(f"matrix shape {matrix.shape} does not fit DiskANN u32 header")

    with path.open("wb") as output:
        output.write(struct.pack("<II", matrix.shape[0], matrix.shape[1]))
        output.write(matrix.tobytes(order="C"))
    print(f"wrote {path} shape={matrix.shape} dtype={matrix.dtype}")


def write_truthset(path: Path, neighbors: np.ndarray, distances: np.ndarray | None) -> None:
    neighbors = np.ascontiguousarray(neighbors, dtype=np.uint32)
    if neighbors.ndim != 2:
        raise ValueError(f"neighbors must be 2-D, got shape {neighbors.shape}")

    if distances is not None:
        distances = np.ascontiguousarray(distances, dtype=np.float32)
        if distances.shape != neighbors.shape:
            raise ValueError(
                f"distances shape {distances.shape} must match neighbors shape {neighbors.shape}"
            )

    with path.open("wb") as output:
        output.write(struct.pack("<II", neighbors.shape[0], neighbors.shape[1]))
        output.write(neighbors.tobytes(order="C"))
        if distances is not None:
            output.write(distances.tobytes(order="C"))
    suffix = "ids+distances" if distances is not None else "ids"
    print(f"wrote {path} shape={neighbors.shape} payload={suffix}")


if __name__ == "__main__":
    main()
