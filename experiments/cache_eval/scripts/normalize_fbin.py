#!/usr/bin/env python3
"""L2-normalize rows of an f32 DiskANN binary matrix."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_metadata(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.read(8)
    if len(header) != 8:
        raise ValueError(f"{path} is too short to contain fbin metadata")
    npoints, dimensions = struct.unpack("<II", header)
    expected_size = 8 + npoints * dimensions * np.dtype("<f4").itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path} has {actual_size} bytes, expected {expected_size} "
            f"for {npoints}x{dimensions} f32 vectors"
        )
    return npoints, dimensions


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --force")

    npoints, dimensions = read_metadata(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(struct.pack("<II", npoints, dimensions))
        output.truncate(8 + npoints * dimensions * 4)

    source = np.memmap(
        args.input,
        mode="r",
        dtype="<f4",
        offset=8,
        shape=(npoints, dimensions),
    )
    destination = np.memmap(
        args.output,
        mode="r+",
        dtype="<f4",
        offset=8,
        shape=(npoints, dimensions),
    )

    zero_norm_rows = 0
    min_norm = float("inf")
    max_norm = 0.0
    for start in range(0, npoints, args.batch_size):
        end = min(start + args.batch_size, npoints)
        batch = np.asarray(source[start:end], dtype=np.float32)
        norms = np.linalg.norm(batch, axis=1)
        zero_mask = norms == 0
        zero_norm_rows += int(zero_mask.sum())
        if np.any(~zero_mask):
            nonzero_norms = norms[~zero_mask]
            min_norm = min(min_norm, float(nonzero_norms.min()))
            max_norm = max(max_norm, float(nonzero_norms.max()))
        normalized = np.zeros_like(batch)
        np.divide(batch, norms[:, None], out=normalized, where=~zero_mask[:, None])
        destination[start:end] = normalized

    destination.flush()
    output_norms = np.linalg.norm(destination, axis=1)
    nonzero_output = output_norms[output_norms > 0]
    max_unit_error = (
        float(np.max(np.abs(nonzero_output - 1.0))) if len(nonzero_output) else 0.0
    )
    print(
        f"normalized {npoints}x{dimensions} vectors from {args.input} to {args.output}; "
        f"input_norm_range=[{min_norm:.6g}, {max_norm:.6g}], "
        f"zero_rows={zero_norm_rows}, max_unit_norm_error={max_unit_error:.3g}"
    )


if __name__ == "__main__":
    main()
