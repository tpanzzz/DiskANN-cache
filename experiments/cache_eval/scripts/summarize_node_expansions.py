#!/usr/bin/env python3
"""Summarize per-node expansion counts from a DiskANN cache-access trace."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--field",
        default="vertex_id",
        help="JSON field containing the expanded node id.",
    )
    return parser.parse_args()


def read_counts(path: Path, field: str, num_nodes: int) -> tuple[np.ndarray, int]:
    counts = np.zeros(num_nodes, dtype=np.uint64)
    malformed = 0
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
                node_id = int(value[field])
                if node_id < 0 or node_id >= num_nodes:
                    raise ValueError(
                        f"node id {node_id} outside [0, {num_nodes})"
                    )
                counts[node_id] += 1
            except Exception as error:
                malformed += 1
                if malformed <= 5:
                    print(
                        f"warning: ignored malformed line {line_number}: {error}",
                        file=sys.stderr,
                    )
    return counts, malformed


def gini(values: np.ndarray) -> float:
    values = np.sort(values.astype(np.float64))
    total = values.sum()
    if total == 0:
        return 0.0
    n = len(values)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * np.dot(ranks, values) / (n * total)) - (n + 1) / n)


def nodes_for_mass(cumulative_mass: np.ndarray, target: float) -> int:
    if len(cumulative_mass) == 0 or cumulative_mass[-1] == 0:
        return 0
    return int(np.searchsorted(cumulative_mass, target, side="left") + 1)


def save_plots(output_dir: Path, sorted_counts: np.ndarray) -> None:
    node_fraction = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    total = sorted_counts.sum()
    cumulative_mass = (
        np.cumsum(sorted_counts, dtype=np.float64) / total
        if total > 0
        else np.zeros(len(sorted_counts), dtype=np.float64)
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(node_fraction, sorted_counts, linewidth=1)
    axis.set_yscale("symlog", linthresh=1)
    axis.set_xlabel("Cumulative fraction of graph nodes, sorted by expansion count")
    axis.set_ylabel("Expansion count")
    axis.set_title("Per-node graph expansion frequency")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "node_expansion_rank_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(node_fraction, cumulative_mass, linewidth=1.5)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_xlabel("Cumulative fraction of graph nodes, sorted by expansion count")
    axis.set_ylabel("Cumulative fraction of all expansions")
    axis.set_title("Expansion concentration curve")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "node_expansion_cumulative_mass.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_nodes <= 0:
        raise ValueError("--num-nodes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts, malformed = read_counts(args.trace, args.field, args.num_nodes)
    order = np.argsort(counts)[::-1]
    sorted_counts = counts[order]
    total_expansions = int(sorted_counts.sum())
    cumulative_mass = (
        np.cumsum(sorted_counts, dtype=np.float64) / total_expansions
        if total_expansions > 0
        else np.zeros(len(sorted_counts), dtype=np.float64)
    )

    frame = pd.DataFrame(
        {
            "rank": np.arange(1, args.num_nodes + 1),
            "node_id": order,
            "expansion_count": sorted_counts,
            "cumulative_node_fraction": np.arange(1, args.num_nodes + 1)
            / args.num_nodes,
            "cumulative_expansion_fraction": cumulative_mass,
        }
    )
    frame.to_csv(args.output_dir / "node_expansion_counts.csv", index=False)
    save_plots(args.output_dir, sorted_counts)

    percentiles = {}
    nonzero = counts[counts > 0]
    for percentile in [50, 90, 95, 99, 99.9]:
        percentiles[str(percentile)] = (
            float(np.percentile(nonzero, percentile)) if len(nonzero) else 0.0
        )

    summary = {
        "trace": str(args.trace.resolve()),
        "num_nodes": args.num_nodes,
        "total_expansions": total_expansions,
        "unique_expanded_nodes": int(np.count_nonzero(counts)),
        "expanded_node_fraction": float(np.count_nonzero(counts) / args.num_nodes),
        "malformed_trace_lines": malformed,
        "max_expansion_count": int(sorted_counts[0]) if len(sorted_counts) else 0,
        "mean_expansions_per_node": float(counts.mean()),
        "gini": gini(counts),
        "expansion_count_percentiles_among_nonzero_nodes": percentiles,
        "nodes_for_50_percent_expansions": nodes_for_mass(cumulative_mass, 0.50),
        "nodes_for_80_percent_expansions": nodes_for_mass(cumulative_mass, 0.80),
        "nodes_for_90_percent_expansions": nodes_for_mass(cumulative_mass, 0.90),
        "nodes_for_95_percent_expansions": nodes_for_mass(cumulative_mass, 0.95),
        "top_1_percent_nodes_expansion_fraction": float(
            sorted_counts[: max(1, math.ceil(args.num_nodes * 0.01))].sum()
            / total_expansions
        )
        if total_expansions
        else 0.0,
        "top_5_percent_nodes_expansion_fraction": float(
            sorted_counts[: max(1, math.ceil(args.num_nodes * 0.05))].sum()
            / total_expansions
        )
        if total_expansions
        else 0.0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    markdown = [
        "# Node expansion distribution",
        "",
        f"- trace: `{args.trace}`",
        f"- graph nodes: {args.num_nodes}",
        f"- total expansions: {total_expansions}",
        f"- unique expanded nodes: {summary['unique_expanded_nodes']}",
        f"- expansion-count Gini: {summary['gini']:.6f}",
        f"- top 1% nodes expansion share: {summary['top_1_percent_nodes_expansion_fraction']:.6f}",
        f"- top 5% nodes expansion share: {summary['top_5_percent_nodes_expansion_fraction']:.6f}",
        f"- nodes covering 50% of expansions: {summary['nodes_for_50_percent_expansions']}",
        f"- nodes covering 90% of expansions: {summary['nodes_for_90_percent_expansions']}",
        "",
        "Generated plots:",
        "",
        "- `node_expansion_rank_curve.png`",
        "- `node_expansion_cumulative_mass.png`",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(markdown) + "\n")
    print(f"wrote expansion summary to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
