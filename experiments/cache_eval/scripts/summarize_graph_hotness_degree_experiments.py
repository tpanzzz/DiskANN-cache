#!/usr/bin/env python3
"""Aggregate graph degree and expansion-hotness experiment results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASETS = ["sift1m", "fashion_mnist", "glove_25", "lastfm_64"]
SOURCES = ["test queries", "all base queries", "1% base sample"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_recall(log_path: Path, search_list: int) -> float:
    for line in log_path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 13 and fields[0].isdigit() and int(fields[0]) == search_list:
            return float(fields[-1])
    raise ValueError(f"could not find L={search_list} result row in {log_path}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    degree_rows = []
    expansion_rows = []
    overlap_rows = []
    hot_degree_rows = []

    for dataset in DATASETS:
        dataset_dir = args.root / dataset
        summary = json.loads((dataset_dir / "analysis/summary.json").read_text())
        config = summary["search_config"]
        search_list = int(config["search_list"])
        test_recall = parse_recall(
            dataset_dir / "search/test_queries.search.log", search_list
        )
        quality = summary["self_query_quality"]
        graph = summary["graph_metadata"]
        graph_degree = summary["graph_degree"]
        overlaps = summary["hot_set_overlap"]
        test_hot_1 = summary["hot_set_degree"]["test queries:top_1"]
        test_correlation = summary["correlations"]["test queries"]["in_degree"]

        dataset_rows.append(
            {
                "dataset_key": dataset,
                "dataset": config["display_name"],
                "graph_nodes": graph["num_nodes"],
                "dimensions": graph["dimensions"],
                "max_degree_R": graph["max_degree"],
                "build_search_list": config.get("build_search_list"),
                "search_list_L": search_list,
                "metric": config["metric"],
                "navigation": config.get("navigation", "pq"),
                "pq_chunks": config["pq_chunks"],
                "threads": config["num_threads"],
                "test_recall_at_10_percent": test_recall,
                "all_base_identity_recall_at_1_percent": quality["all_base_queries"][
                    "identity_recall_at_1"
                ]
                * 100,
                "all_base_distance_equivalent_recall_at_1_percent": quality[
                    "all_base_queries"
                ]["distance_equivalent_recall_at_1"]
                * 100,
                "sample_identity_recall_at_1_percent": quality[
                    "sampled_base_queries"
                ]["identity_recall_at_1"]
                * 100,
                "sample_distance_equivalent_recall_at_1_percent": quality[
                    "sampled_base_queries"
                ]["distance_equivalent_recall_at_1"]
                * 100,
                "test_top_1_expansion_mass_percent": summary["expansion"][
                    "test queries"
                ]["top_1_percent_expansion_mass"]
                * 100,
                "test_top_5_expansion_mass_percent": summary["expansion"][
                    "test queries"
                ]["top_5_percent_expansion_mass"]
                * 100,
                "base_test_top_1_overlap_percent": overlaps["base_test_top_1"][
                    "overlap_coefficient"
                ]
                * 100,
                "base_test_top_5_overlap_percent": overlaps["base_test_top_5"][
                    "overlap_coefficient"
                ]
                * 100,
                "sample_test_top_1_overlap_percent": overlaps["sample_test_top_1"][
                    "overlap_coefficient"
                ]
                * 100,
                "sample_test_top_5_overlap_percent": overlaps["sample_test_top_5"][
                    "overlap_coefficient"
                ]
                * 100,
                "test_hot_top_1_in_degree_enrichment": test_hot_1["in_degree"][
                    "mean_enrichment_over_graph"
                ],
                "test_in_degree_expansion_spearman": test_correlation[
                    "degree_expansion_spearman_all_nodes"
                ],
            }
        )

        for direction in ["out_degree", "in_degree"]:
            degree_rows.append(
                {
                    "dataset_key": dataset,
                    "dataset": config["display_name"],
                    "direction": direction,
                    **graph_degree[direction],
                }
            )

        for source in SOURCES:
            expansion_rows.append(
                {
                    "dataset_key": dataset,
                    "dataset": config["display_name"],
                    "query_source": source,
                    **summary["expansion"][source],
                }
            )
            for top_label in ["top_1", "top_5"]:
                value = summary["hot_set_degree"][f"{source}:{top_label}"]
                hot_degree_rows.append(
                    {
                        "dataset_key": dataset,
                        "dataset": config["display_name"],
                        "query_source": source,
                        "hot_set": top_label,
                        "node_count": value["node_count"],
                        "out_degree_mean": value["out_degree"]["mean"],
                        "in_degree_mean": value["in_degree"]["mean"],
                        "in_degree_median": value["in_degree"]["median"],
                        "in_degree_p95": value["in_degree"]["p95"],
                        "in_degree_max": value["in_degree"]["max"],
                        "in_degree_mean_enrichment": value["in_degree"][
                            "mean_enrichment_over_graph"
                        ],
                    }
                )

        for comparison, value in overlaps.items():
            overlap_rows.append(
                {
                    "dataset_key": dataset,
                    "dataset": config["display_name"],
                    "comparison": comparison,
                    **value,
                }
            )

    dataset_frame = pd.DataFrame(dataset_rows)
    degree_frame = pd.DataFrame(degree_rows)
    expansion_frame = pd.DataFrame(expansion_rows)
    overlap_frame = pd.DataFrame(overlap_rows)
    hot_degree_frame = pd.DataFrame(hot_degree_rows)

    dataset_frame.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    degree_frame.to_csv(args.output_dir / "degree_summary.csv", index=False)
    expansion_frame.to_csv(args.output_dir / "expansion_summary.csv", index=False)
    overlap_frame.to_csv(args.output_dir / "hot_set_overlap.csv", index=False)
    hot_degree_frame.to_csv(args.output_dir / "hot_set_degree_summary.csv", index=False)

    source_paths = [
        "diskann-disk/src/search/provider/cached_disk_vertex_provider.rs",
        "experiments/cache_eval/scripts/prepare_base_query_workloads.py",
        "experiments/cache_eval/scripts/validate_self_query_results.py",
        "experiments/cache_eval/scripts/analyze_graph_hotness_degree.py",
        "experiments/cache_eval/scripts/run_graph_hotness_degree_experiments.sh",
        "experiments/cache_eval/scripts/summarize_graph_hotness_degree_experiments.py",
        "target/release/search_disk_index",
    ]
    source_rows = []
    for relative_path in source_paths:
        path = args.repo_root / relative_path
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(8 * 1024 * 1024):
                digest.update(chunk)
        source_rows.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    pd.DataFrame(source_rows).to_csv(
        args.output_dir / "source_files_sha256.csv", index=False
    )

    labels = dataset_frame["dataset"].tolist()
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.35
    axis.bar(
        x - width / 2,
        dataset_frame["test_top_1_expansion_mass_percent"],
        width,
        label="Top 1% nodes",
        color="#35618f",
    )
    axis.bar(
        x + width / 2,
        dataset_frame["test_top_5_expansion_mass_percent"],
        width,
        label="Top 5% nodes",
        color="#b07a55",
    )
    axis.set_xticks(x, labels, rotation=12, ha="right")
    axis.set_ylabel("Share of test-query expansions (%)")
    axis.set_title("Test-query expansion concentration")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "test_expansion_concentration.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    columns = [
        "base_test_top_1_overlap_percent",
        "base_test_top_5_overlap_percent",
        "sample_test_top_1_overlap_percent",
        "sample_test_top_5_overlap_percent",
    ]
    names = ["All base/top 1%", "All base/top 5%", "1% sample/top 1%", "1% sample/top 5%"]
    colors = ["#35618f", "#5686a6", "#8b5a3c", "#b07a55"]
    width = 0.19
    for index, (column, name, color) in enumerate(zip(columns, names, colors)):
        axis.bar(
            x + (index - 1.5) * width,
            dataset_frame[column],
            width,
            label=name,
            color=color,
        )
    axis.set_xticks(x, labels, rotation=12, ha="right")
    axis.set_ylabel("Exact hot-set overlap with test queries (%)")
    axis.set_title("How well base-derived workloads predict test hot nodes")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(args.output_dir / "hot_set_overlap_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(
        labels,
        dataset_frame["test_hot_top_1_in_degree_enrichment"],
        color=["#35618f", "#4f7a4e", "#8b5a3c", "#7a5a8f"],
    )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.bar_label(bars, fmt="%.2fx", padding=3)
    axis.set_ylabel("Mean in-degree enrichment")
    axis.set_title("In-degree enrichment of test expansion top 1% nodes")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=12)
    figure.tight_layout()
    figure.savefig(args.output_dir / "test_hot_in_degree_enrichment.png", dpi=180)
    plt.close(figure)

    print(f"wrote aggregate summaries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
