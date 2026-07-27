#!/usr/bin/env python3
"""Validate self-query results by both exact id and distance equivalence."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-file", type=Path, required=True)
    parser.add_argument("--all-result", type=Path, required=True)
    parser.add_argument("--sample-result", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path, required=True)
    parser.add_argument(
        "--metric", choices=["l2", "cosine", "cosinenormalized"], required=True
    )
    parser.add_argument("--distance-tolerance", type=float, default=1e-6)
    parser.add_argument("--required-recall", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_fbin(path: Path) -> tuple[np.memmap, int, int]:
    with path.open("rb") as file:
        rows, dimensions = struct.unpack("<II", file.read(8))
    matrix = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=8,
        shape=(rows, dimensions),
    )
    return matrix, rows, dimensions


def read_result_ids(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        rows, dimensions = struct.unpack("<II", file.read(8))
        ids = np.fromfile(file, dtype="<u4", count=rows * dimensions)
    if dimensions != 1:
        raise ValueError(f"expected one result per query in {path}, found {dimensions}")
    if len(ids) != rows:
        raise ValueError(f"truncated result file: {path}")
    return ids


def read_sample_ids(path: Path) -> np.ndarray:
    values = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            values.append(int(row["node_id"]))
    return np.asarray(values, dtype=np.uint32)


def validate_workload(
    base: np.ndarray,
    expected_ids: np.ndarray,
    returned_ids: np.ndarray,
    metric: str,
    tolerance: float,
) -> dict:
    if len(expected_ids) != len(returned_ids):
        raise ValueError(
            f"query/result row mismatch: {len(expected_ids)} versus {len(returned_ids)}"
        )
    if np.any(returned_ids >= len(base)):
        raise ValueError("result contains a node id outside the base matrix")

    distances = np.empty(len(expected_ids), dtype=np.float32)
    chunk_size = 100_000
    for start in range(0, len(expected_ids), chunk_size):
        stop = min(len(expected_ids), start + chunk_size)
        query = np.asarray(base[expected_ids[start:stop]], dtype=np.float32)
        result = np.asarray(base[returned_ids[start:stop]], dtype=np.float32)
        if metric == "l2":
            delta = query - result
            distances[start:stop] = np.einsum("ij,ij->i", delta, delta)
        else:
            dot = np.einsum("ij,ij->i", query, result)
            norms = np.linalg.norm(query, axis=1) * np.linalg.norm(result, axis=1)
            distances[start:stop] = 1.0 - np.divide(
                dot,
                norms,
                out=np.full_like(dot, np.inf),
                where=norms > 0,
            )

    return {
        "queries": len(expected_ids),
        "identity_recall_at_1": float(np.mean(returned_ids == expected_ids)),
        "distance_equivalent_recall_at_1": float(np.mean(distances <= tolerance)),
        "distance_tolerance": tolerance,
        "distance_p50": float(np.percentile(distances, 50)),
        "distance_p95": float(np.percentile(distances, 95)),
        "distance_p99": float(np.percentile(distances, 99)),
        "distance_p999": float(np.percentile(distances, 99.9)),
        "distance_max": float(distances.max()),
    }


def main() -> int:
    args = parse_args()
    if args.distance_tolerance < 0:
        raise ValueError("--distance-tolerance must be non-negative")

    base, rows, dimensions = read_fbin(args.base_file)
    all_returned = read_result_ids(args.all_result)
    sample_returned = read_result_ids(args.sample_result)
    sample_ids = read_sample_ids(args.sample_ids)

    summary = {
        "base_file": str(args.base_file.resolve()),
        "base_rows": rows,
        "dimensions": dimensions,
        "metric": args.metric,
        "required_distance_equivalent_recall": args.required_recall,
        "all_base_queries": validate_workload(
            base,
            np.arange(rows, dtype=np.uint32),
            all_returned,
            args.metric,
            args.distance_tolerance,
        ),
        "sampled_base_queries": validate_workload(
            base,
            sample_ids,
            sample_returned,
            args.metric,
            args.distance_tolerance,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if any(
        summary[name]["distance_equivalent_recall_at_1"] < args.required_recall
        for name in ["all_base_queries", "sampled_base_queries"]
    ):
        raise SystemExit("distance-equivalent Recall@1 is below the required threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
