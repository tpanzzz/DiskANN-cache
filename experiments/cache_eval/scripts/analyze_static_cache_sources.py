#!/usr/bin/env python3
"""Build and compare equal-capacity static-cache node sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mmap
import struct
from collections import deque
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE_LABELS = {
    "bfs": "Medoid BFS",
    "all_base_hot": "All-base hot",
    "sample_base_hot": "1% base-sample hot",
    "test_hot": "Test-query hot (oracle)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument("--test-counts", type=Path, required=True)
    parser.add_argument("--base-counts", type=Path, required=True)
    parser.add_argument("--sample-counts", type=Path, required=True)
    parser.add_argument("--cache-fraction", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_graph_header(index_file: Path, expected_dimensions: int) -> dict:
    with index_file.open("rb") as file:
        serialized = file.read(104)
    if len(serialized) != 104 or struct.unpack_from("<II", serialized, 0) != (96, 1):
        raise ValueError(f"invalid DiskANN graph header in {index_file}")
    values = struct.unpack_from("<10Q", serialized, 8)
    (
        num_nodes,
        dimensions,
        medoid,
        node_len,
        nodes_per_block,
        _frozen_count,
        _frozen_location,
        _append_reorder_data,
        _file_size,
        associated_data_length,
    ) = values
    block_size = struct.unpack_from("<Q", serialized, 88)[0] or 4096
    if dimensions != expected_dimensions:
        raise ValueError(
            f"index dimensions {dimensions} do not match expected {expected_dimensions}"
        )
    vector_bytes = dimensions * 4
    max_degree = (node_len - vector_bytes - associated_data_length) // 4 - 1
    return {
        "num_nodes": int(num_nodes),
        "dimensions": int(dimensions),
        "medoid": int(medoid),
        "node_len": int(node_len),
        "nodes_per_block": int(nodes_per_block),
        "block_size": int(block_size),
        "vector_bytes": int(vector_bytes),
        "max_degree": int(max_degree),
    }


def node_record_offset(node_id: int, metadata: dict) -> int:
    nodes_per_block = metadata["nodes_per_block"]
    if nodes_per_block:
        return (
            (1 + node_id // nodes_per_block) * metadata["block_size"]
            + (node_id % nodes_per_block) * metadata["node_len"]
        )
    blocks_per_node = math.ceil(metadata["node_len"] / metadata["block_size"])
    return (1 + node_id * blocks_per_node) * metadata["block_size"]


def bfs_nodes(index_file: Path, metadata: dict, capacity: int) -> np.ndarray:
    queue: deque[int] = deque([metadata["medoid"]])
    visited = {metadata["medoid"]}
    selected: list[int] = []
    with index_file.open("rb") as file:
        mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        while queue and len(selected) < capacity:
            node_id = queue.popleft()
            selected.append(node_id)
            adjacency_offset = (
                node_record_offset(node_id, metadata) + metadata["vector_bytes"]
            )
            degree = struct.unpack_from("<I", mapped, adjacency_offset)[0]
            if degree > metadata["max_degree"]:
                raise ValueError(
                    f"node {node_id} degree {degree} exceeds {metadata['max_degree']}"
                )
            neighbors = struct.unpack_from(
                f"<{degree}I", mapped, adjacency_offset + 4
            )
            for neighbor in neighbors:
                if neighbor >= metadata["num_nodes"]:
                    raise ValueError(f"node {node_id} has invalid neighbor {neighbor}")
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        mapped.close()
    if len(selected) != capacity:
        raise ValueError(f"BFS reached only {len(selected)} of {capacity} requested nodes")
    return np.asarray(selected, dtype=np.uint32)


def read_counts(path: Path, num_nodes: int) -> np.ndarray:
    counts = np.zeros(num_nodes, dtype=np.uint64)
    seen: set[int] = set()
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["node_id", "expansion_count"]:
            raise ValueError(f"unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            node_id = int(row["node_id"])
            count = int(row["expansion_count"])
            if not 0 <= node_id < num_nodes or node_id in seen or count <= 0:
                raise ValueError(f"invalid count row for node {node_id} in {path}")
            seen.add(node_id)
            counts[node_id] = count
    return counts


def top_nodes(counts: np.ndarray, capacity: int) -> np.ndarray:
    node_ids = np.arange(len(counts), dtype=np.int64)
    order = np.lexsort((node_ids, -counts.astype(np.int64)))
    return order[:capacity].astype(np.uint32)


def write_node_set(path: Path, node_ids: np.ndarray, source_counts: np.ndarray | None) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["node_id", "source_rank", "source_expansion_count"])
        for rank, node_id in enumerate(node_ids, start=1):
            writer.writerow(
                [
                    int(node_id),
                    rank,
                    "" if source_counts is None else int(source_counts[node_id]),
                ]
            )


def pairwise_rows(
    sources: dict[str, np.ndarray], test_counts: np.ndarray
) -> list[dict]:
    total_test_expansions = int(test_counts.sum())
    rows = []
    for left_name, right_name in combinations(sources, 2):
        left = sources[left_name]
        right = sources[right_name]
        intersection = np.intersect1d(left, right, assume_unique=True)
        union_size = len(left) + len(right) - len(intersection)
        intersection_mass = int(test_counts[intersection].sum())
        left_mass = int(test_counts[left].sum())
        right_mass = int(test_counts[right].sum())
        rows.append(
            {
                "left_source": left_name,
                "right_source": right_name,
                "left_nodes": len(left),
                "right_nodes": len(right),
                "intersection_nodes": len(intersection),
                "overlap_coefficient_percent": 100 * len(intersection) / min(len(left), len(right)),
                "jaccard_percent": 100 * len(intersection) / union_size,
                "intersection_test_expansions": intersection_mass,
                "intersection_test_expansion_share_percent": 100 * intersection_mass / total_test_expansions,
                "intersection_share_of_left_test_expansions_percent": 100 * intersection_mass / left_mass if left_mass else 0,
                "intersection_share_of_right_test_expansions_percent": 100 * intersection_mass / right_mass if right_mass else 0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir: Path, dataset: str, sources: dict[str, np.ndarray], rows: list[dict], test_counts: np.ndarray) -> None:
    names = list(sources)
    overlap = np.eye(len(names)) * 100
    mass = np.zeros((len(names), len(names)))
    total_mass = test_counts.sum()
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            intersection = np.intersect1d(sources[left], sources[right], assume_unique=True)
            if i != j:
                overlap[i, j] = 100 * len(intersection) / len(sources[left])
            mass[i, j] = 100 * test_counts[intersection].sum() / total_mass

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    labels = [SOURCE_LABELS[name] for name in names]
    for axis, matrix, title, fmt in [
        (axes[0], overlap, "Node-set overlap coefficient (%)", ".1f"),
        (axes[1], mass, "Test expansion mass in intersection (%)", ".2f"),
    ]:
        image = axis.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(title)
        for row in range(len(labels)):
            for column in range(len(labels)):
                color = "white" if matrix[row, column] < 55 else "black"
                axis.text(column, row, format(matrix[row, column], fmt), ha="center", va="center", color=color)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"{dataset}: equal-capacity static cache sources")
    figure.tight_layout()
    figure.savefig(output_dir / "pairwise_overlap_and_test_expansion_mass.png", dpi=180)
    plt.close(figure)

    coverage = [100 * test_counts[sources[name]].sum() / total_mass for name in names]
    figure, axis = plt.subplots(figsize=(8.5, 5))
    bars = axis.bar(labels, coverage, color=["#35618f", "#568b78", "#b06d3c", "#8a5b9e"])
    axis.bar_label(bars, fmt="%.2f%%", padding=3)
    axis.set_ylabel("Share of all test-query expansions (%)")
    axis.set_title(f"{dataset}: test expansion coverage of each static source")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_dir / "source_test_expansion_coverage.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not 0 < args.cache_fraction <= 1:
        raise ValueError("cache fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_graph_header(args.index_file, args.dimensions)
    capacity = max(1, math.ceil(metadata["num_nodes"] * args.cache_fraction))
    test_counts = read_counts(args.test_counts, metadata["num_nodes"])
    base_counts = read_counts(args.base_counts, metadata["num_nodes"])
    sample_counts = read_counts(args.sample_counts, metadata["num_nodes"])
    sources = {
        "bfs": bfs_nodes(args.index_file, metadata, capacity),
        "all_base_hot": top_nodes(base_counts, capacity),
        "sample_base_hot": top_nodes(sample_counts, capacity),
        "test_hot": top_nodes(test_counts, capacity),
    }
    source_counts = {
        "bfs": None,
        "all_base_hot": base_counts,
        "sample_base_hot": sample_counts,
        "test_hot": test_counts,
    }
    for name, ids in sources.items():
        write_node_set(args.output_dir / f"{name}.nodes.csv", ids, source_counts[name])

    pairwise = pairwise_rows(sources, test_counts)
    write_csv(args.output_dir / "pairwise_overlap.csv", pairwise)
    total_test_expansions = int(test_counts.sum())
    coverage = [
        {
            "source": name,
            "nodes": len(ids),
            "test_expansions": int(test_counts[ids].sum()),
            "test_expansion_share_percent": 100 * test_counts[ids].sum() / total_test_expansions,
        }
        for name, ids in sources.items()
    ]
    write_csv(args.output_dir / "source_test_expansion_coverage.csv", coverage)
    save_plots(args.output_dir, args.dataset, sources, pairwise, test_counts)

    inputs = [args.index_file, args.test_counts, args.base_counts, args.sample_counts]
    manifest = {
        "dataset": args.dataset,
        "cache_fraction": args.cache_fraction,
        "cache_capacity_nodes": capacity,
        "graph_metadata": metadata,
        "inputs": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
        "outputs": {
            name: str((args.output_dir / f"{name}.nodes.csv").resolve())
            for name in sources
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.dataset} static-cache source analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
