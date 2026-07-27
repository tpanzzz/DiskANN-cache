#!/usr/bin/env python3
"""Analyze DiskANN graph degree, expansion hot sets, and cross-workload overlap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mmap
import struct
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument("--test-counts", type=Path, required=True)
    parser.add_argument("--base-counts", type=Path, required=True)
    parser.add_argument("--sample-counts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--self-quality", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, action="append", default=[])
    return parser.parse_args()


def read_graph_degrees(
    index_file: Path, expected_dimensions: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    with index_file.open("rb") as file:
        serialized_header = file.read(104)
    if len(serialized_header) != 104:
        raise ValueError(f"{index_file} is too short to contain a GraphHeader")
    header_rows, header_columns = struct.unpack_from("<II", serialized_header, 0)
    if (header_rows, header_columns) != (96, 1):
        raise ValueError(
            f"unexpected serialized GraphHeader shape "
            f"{header_rows}x{header_columns} in {index_file}"
        )
    header = serialized_header[8:]

    metadata_values = struct.unpack_from("<10Q", header, 0)
    (
        num_nodes,
        dimensions,
        medoid,
        node_len,
        nodes_per_block,
        frozen_count,
        frozen_location,
        _append_reorder_data,
        declared_file_size,
        associated_data_length,
    ) = metadata_values
    block_size = struct.unpack_from("<Q", header, 80)[0] or 4096
    layout_major, layout_minor = struct.unpack_from("<II", header, 88)

    if dimensions != expected_dimensions:
        raise ValueError(
            f"index dimensions {dimensions} do not match expected {expected_dimensions}"
        )
    vector_bytes = dimensions * np.dtype("<f4").itemsize
    if vector_bytes + 4 > node_len:
        raise ValueError("node record is too short to contain vector and degree")
    max_degree = (node_len - vector_bytes - associated_data_length) // 4 - 1

    ids = np.arange(num_nodes, dtype=np.uint64)
    if nodes_per_block:
        offsets = (
            (1 + ids // nodes_per_block) * block_size
            + (ids % nodes_per_block) * node_len
            + vector_bytes
        )
    else:
        blocks_per_node = math.ceil(node_len / block_size)
        offsets = (1 + ids * blocks_per_node) * block_size + vector_bytes

    out_degrees = np.empty(num_nodes, dtype=np.uint32)
    in_degrees = np.zeros(num_nodes, dtype=np.uint64)
    with index_file.open("rb") as file:
        mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        raw = np.frombuffer(mapped, dtype=np.uint8)
        chunk_size = 1_000_000
        for start in range(0, num_nodes, chunk_size):
            stop = min(num_nodes, start + chunk_size)
            chunk_offsets = offsets[start:stop]
            out_degrees[start:stop] = (
                raw[chunk_offsets].astype(np.uint32)
                | (raw[chunk_offsets + 1].astype(np.uint32) << 8)
                | (raw[chunk_offsets + 2].astype(np.uint32) << 16)
                | (raw[chunk_offsets + 3].astype(np.uint32) << 24)
            )

        edge_chunk_size = 50_000
        neighbor_positions = np.arange(max_degree, dtype=np.uint64)
        for start in range(0, num_nodes, edge_chunk_size):
            stop = min(num_nodes, start + edge_chunk_size)
            chunk_degrees = out_degrees[start:stop]
            adjacency_offsets = offsets[start:stop] + 4
            neighbor_offsets = adjacency_offsets[:, None] + 4 * neighbor_positions
            active = neighbor_positions < chunk_degrees[:, None]
            active_offsets = neighbor_offsets[active]
            neighbors = (
                raw[active_offsets].astype(np.uint32)
                | (raw[active_offsets + 1].astype(np.uint32) << 8)
                | (raw[active_offsets + 2].astype(np.uint32) << 16)
                | (raw[active_offsets + 3].astype(np.uint32) << 24)
            )
            if np.any(neighbors >= num_nodes):
                raise ValueError("adjacency list contains an out-of-range node id")
            in_degrees += np.bincount(neighbors, minlength=num_nodes).astype(np.uint64)
        del raw
        mapped.close()

    if np.any(out_degrees > max_degree):
        bad_id = int(np.flatnonzero(out_degrees > max_degree)[0])
        raise ValueError(
            f"node {bad_id} has degree {out_degrees[bad_id]}, "
            f"above header maximum {max_degree}"
        )
    if int(out_degrees.sum()) != int(in_degrees.sum()):
        raise ValueError("in-degree and out-degree edge totals do not match")

    metadata = {
        "num_nodes": int(num_nodes),
        "dimensions": int(dimensions),
        "medoid": int(medoid),
        "node_len": int(node_len),
        "nodes_per_block": int(nodes_per_block),
        "block_size": int(block_size),
        "max_degree": int(max_degree),
        "frozen_count": int(frozen_count),
        "frozen_location": int(frozen_location),
        "declared_file_size": int(declared_file_size),
        "actual_file_size": index_file.stat().st_size,
        "associated_data_length": int(associated_data_length),
        "layout_version": f"{layout_major}.{layout_minor}",
    }
    return out_degrees, in_degrees.astype(np.uint32), metadata


def read_sparse_counts(path: Path, num_nodes: int) -> np.ndarray:
    counts = np.zeros(num_nodes, dtype=np.uint64)
    seen: set[int] = set()
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["node_id", "expansion_count"]:
            raise ValueError(f"unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            node_id = int(row["node_id"])
            count = int(row["expansion_count"])
            if node_id < 0 or node_id >= num_nodes:
                raise ValueError(f"node id {node_id} outside [0, {num_nodes}) in {path}")
            if node_id in seen:
                raise ValueError(f"duplicate node id {node_id} in {path}")
            if count <= 0:
                raise ValueError(f"non-positive expansion count for node {node_id} in {path}")
            seen.add(node_id)
            counts[node_id] = count
    return counts


def descending_order(values: np.ndarray) -> np.ndarray:
    node_ids = np.arange(len(values), dtype=np.int64)
    return np.lexsort((node_ids, -values.astype(np.int64)))


def gini(values: np.ndarray) -> float:
    sorted_values = np.sort(values.astype(np.float64))
    total = sorted_values.sum()
    if total == 0:
        return 0.0
    ranks = np.arange(1, len(sorted_values) + 1, dtype=np.float64)
    return float(
        2 * np.dot(ranks, sorted_values) / (len(sorted_values) * total)
        - (len(sorted_values) + 1) / len(sorted_values)
    )


def nodes_for_mass(sorted_counts: np.ndarray, fraction: float) -> int:
    total = int(sorted_counts.sum())
    if total == 0:
        return 0
    cumulative = np.cumsum(sorted_counts, dtype=np.uint64)
    return int(np.searchsorted(cumulative, total * fraction, side="left") + 1)


def top_ids(counts: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, math.ceil(len(counts) * fraction))
    return descending_order(counts)[:count]


def expansion_summary(counts: np.ndarray) -> dict:
    order = descending_order(counts)
    sorted_counts = counts[order]
    total = int(sorted_counts.sum())
    top_1_count = max(1, math.ceil(len(counts) * 0.01))
    top_5_count = max(1, math.ceil(len(counts) * 0.05))
    return {
        "total_expansions": total,
        "unique_expanded_nodes": int(np.count_nonzero(counts)),
        "expanded_node_fraction": float(np.count_nonzero(counts) / len(counts)),
        "max_expansion_count": int(sorted_counts[0]),
        "mean_expansions_per_graph_node": float(counts.mean()),
        "gini": gini(counts),
        "top_1_percent_expansion_mass": (
            float(sorted_counts[:top_1_count].sum() / total) if total else 0.0
        ),
        "top_5_percent_expansion_mass": (
            float(sorted_counts[:top_5_count].sum() / total) if total else 0.0
        ),
        "nodes_for_50_percent_mass": nodes_for_mass(sorted_counts, 0.50),
        "nodes_for_90_percent_mass": nodes_for_mass(sorted_counts, 0.90),
        "nodes_for_95_percent_mass": nodes_for_mass(sorted_counts, 0.95),
    }


def hot_degree_summary(
    counts: np.ndarray,
    out_degrees: np.ndarray,
    in_degrees: np.ndarray,
    fraction: float,
) -> tuple[dict, np.ndarray]:
    ids = top_ids(counts, fraction)
    selected_out = out_degrees[ids]
    selected_in = in_degrees[ids]
    return (
        {
            "node_count": len(ids),
            "out_degree": degree_summary(selected_out),
            "in_degree": {
                **degree_summary(selected_in),
                "mean_enrichment_over_graph": float(
                    selected_in.mean() / in_degrees.mean()
                ),
            },
        },
        ids,
    )


def overlap(left: np.ndarray, right: np.ndarray) -> dict:
    intersection = len(np.intersect1d(left, right, assume_unique=True))
    union = len(left) + len(right) - intersection
    return {
        "intersection_nodes": intersection,
        "left_nodes": len(left),
        "right_nodes": len(right),
        "overlap_coefficient": intersection / min(len(left), len(right)),
        "jaccard": intersection / union,
    }


def degree_expansion_correlations(degrees: np.ndarray, counts: np.ndarray) -> dict:
    if np.ptp(degrees) == 0 or np.ptp(counts) == 0:
        return {
            "degree_expansion_spearman_all_nodes": None,
            "degree_log1p_expansion_pearson_all_nodes": None,
        }
    return {
        "degree_expansion_spearman_all_nodes": float(
            spearmanr(degrees, counts).statistic
        ),
        "degree_log1p_expansion_pearson_all_nodes": float(
            np.corrcoef(degrees, np.log1p(counts))[0, 1]
        ),
    }


def degree_summary(degrees: np.ndarray) -> dict:
    return {
        "mean": float(degrees.mean()),
        "median": float(np.median(degrees)),
        "p95": float(np.percentile(degrees, 95)),
        "p99": float(np.percentile(degrees, 99)),
        "min": int(degrees.min()),
        "max": int(degrees.max()),
    }


def sampled_curve(values: np.ndarray, max_points: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= max_points:
        indices = np.arange(len(values))
    else:
        indices = np.unique(np.linspace(0, len(values) - 1, max_points).astype(int))
    ranks = (indices + 1) / len(values) * 100
    return ranks, values[indices]


def save_top_set(
    path: Path,
    ids: np.ndarray,
    counts: np.ndarray,
    out_degrees: np.ndarray,
    in_degrees: np.ndarray,
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["expansion_rank", "node_id", "expansion_count", "out_degree", "in_degree"]
        )
        for rank, node_id in enumerate(ids, start=1):
            writer.writerow(
                [
                    rank,
                    int(node_id),
                    int(counts[node_id]),
                    int(out_degrees[node_id]),
                    int(in_degrees[node_id]),
                ]
            )


def save_plots(
    output_dir: Path,
    dataset: str,
    out_degrees: np.ndarray,
    in_degrees: np.ndarray,
    counts_by_source: dict[str, np.ndarray],
    test_hot_ids: dict[str, np.ndarray],
    overlaps: dict[str, dict],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, degrees, title, ylabel in [
        (axes[0], out_degrees, "Stored out-degree", "Out-degree"),
        (axes[1], in_degrees, "Derived in-degree", "In-degree"),
    ]:
        degree_order = descending_order(degrees)
        degree_ranks, sorted_degrees = sampled_curve(degrees[degree_order])
        axis.plot(degree_ranks, sorted_degrees, linewidth=1.4)
        axis.set_xlabel("Cumulative node rank by descending degree (%)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1].set_yscale("symlog", linthresh=1)
    figure.suptitle(f"{dataset}: graph degree rank curves")
    figure.tight_layout()
    figure.savefig(output_dir / "experiment1_graph_degree_rank.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, degrees, title, ylabel in [
        (axes[0], out_degrees, "Stored out-degree", "Out-degree"),
        (axes[1], in_degrees, "Derived in-degree", "In-degree"),
    ]:
        for label, ids in test_hot_ids.items():
            selected_degrees = np.sort(degrees[ids])[::-1]
            ranks, values = sampled_curve(selected_degrees)
            axis.plot(ranks, values, linewidth=1.4, label=label)
        axis.set_xlabel("Cumulative rank within selected hot nodes (%)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].set_yscale("symlog", linthresh=1)
    figure.suptitle(f"{dataset}: degree of test-query expansion hot sets")
    figure.tight_layout()
    figure.savefig(output_dir / "experiment2_test_hot_node_degree_rank.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for source, counts in counts_by_source.items():
        sorted_counts = counts[descending_order(counts)]
        ranks, values = sampled_curve(sorted_counts)
        axis.plot(ranks, values, linewidth=1.3, label=source)
    axis.set_yscale("symlog", linthresh=1)
    axis.set_xlabel("Cumulative node rank by descending expansion count (%)")
    axis.set_ylabel("Expansion count")
    axis.set_title(f"{dataset}: expansion rank curves")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "experiments3_4_expansion_rank.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for source, counts in counts_by_source.items():
        sorted_counts = counts[descending_order(counts)]
        total = int(sorted_counts.sum())
        cumulative = (
            np.cumsum(sorted_counts, dtype=np.float64) / total
            if total
            else np.zeros(len(sorted_counts))
        )
        ranks, values = sampled_curve(cumulative)
        axis.plot(ranks, values * 100, linewidth=1.3, label=source)
    axis.set_xlabel("Cumulative node rank by descending expansion count (%)")
    axis.set_ylabel("Cumulative expansion mass (%)")
    axis.set_title(f"{dataset}: expansion concentration")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "experiments3_4_expansion_mass.png", dpi=180)
    plt.close(figure)

    labels = ["Base vs test\ntop 1%", "Base vs test\ntop 5%", "Sample vs test\ntop 1%", "Sample vs test\ntop 5%"]
    keys = ["base_test_top_1", "base_test_top_5", "sample_test_top_1", "sample_test_top_5"]
    values = [overlaps[key]["overlap_coefficient"] * 100 for key in keys]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=["#35618f", "#5686a6", "#8b5a3c", "#b07a55"])
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Exact-set overlap (%)")
    axis.set_title(f"{dataset}: expansion hot-set overlap with test queries")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "experiments3_4_hot_set_overlap.png", dpi=180)
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_degrees, in_degrees, graph_metadata = read_graph_degrees(
        args.index_file, args.dimensions
    )
    np.savez_compressed(
        args.output_dir / "node_degrees_by_id.npz",
        out_degree=out_degrees,
        in_degree=in_degrees,
    )
    counts_by_source = {
        "test queries": read_sparse_counts(args.test_counts, len(out_degrees)),
        "all base queries": read_sparse_counts(args.base_counts, len(out_degrees)),
        "1% base sample": read_sparse_counts(args.sample_counts, len(out_degrees)),
    }

    expansion_summaries = {
        source: expansion_summary(counts) for source, counts in counts_by_source.items()
    }
    hot_degree_summaries: dict[str, dict] = {}
    hot_ids: dict[str, dict[str, np.ndarray]] = {}
    for source, counts in counts_by_source.items():
        hot_ids[source] = {}
        for label, fraction in [("top_1", 0.01), ("top_5", 0.05)]:
            summary, ids = hot_degree_summary(
                counts, out_degrees, in_degrees, fraction
            )
            hot_degree_summaries[f"{source}:{label}"] = summary
            hot_ids[source][label] = ids
            file_source = source.replace(" ", "_").replace("%", "percent")
            save_top_set(
                args.output_dir / f"{file_source}.{label}.csv",
                ids,
                counts,
                out_degrees,
                in_degrees,
            )

    overlaps = {
        "base_test_top_1": overlap(
            hot_ids["all base queries"]["top_1"], hot_ids["test queries"]["top_1"]
        ),
        "base_test_top_5": overlap(
            hot_ids["all base queries"]["top_5"], hot_ids["test queries"]["top_5"]
        ),
        "sample_test_top_1": overlap(
            hot_ids["1% base sample"]["top_1"], hot_ids["test queries"]["top_1"]
        ),
        "sample_test_top_5": overlap(
            hot_ids["1% base sample"]["top_5"], hot_ids["test queries"]["top_5"]
        ),
    }

    correlations = {
        source: {
            "out_degree": degree_expansion_correlations(out_degrees, counts),
            "in_degree": degree_expansion_correlations(in_degrees, counts),
        }
        for source, counts in counts_by_source.items()
    }
    graph_degree_summary = {
        "out_degree": degree_summary(out_degrees),
        "in_degree": degree_summary(in_degrees),
    }
    search_config = json.loads(args.search_config.read_text())
    self_query_quality = json.loads(args.self_quality.read_text())
    summary = {
        "dataset": args.dataset,
        "graph_metadata": graph_metadata,
        "graph_degree": graph_degree_summary,
        "search_config": search_config,
        "self_query_quality": self_query_quality,
        "expansion": expansion_summaries,
        "hot_set_degree": hot_degree_summaries,
        "hot_set_overlap": overlaps,
        "correlations": correlations,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save_plots(
        args.output_dir,
        args.dataset,
        out_degrees,
        in_degrees,
        counts_by_source,
        {
            "Test expansion top 1%": hot_ids["test queries"]["top_1"],
            "Test expansion top 5%": hot_ids["test queries"]["top_5"],
        },
        overlaps,
    )

    provenance_files = [
        args.index_file,
        args.test_counts,
        args.base_counts,
        args.sample_counts,
        args.search_config,
        args.self_quality,
        *args.input_file,
    ]
    provenance = {
        "dataset": args.dataset,
        "files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in provenance_files
        ],
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"wrote graph-hotness analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
