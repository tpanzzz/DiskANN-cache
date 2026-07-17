#!/usr/bin/env python3
"""Aggregate per-dataset vector-pattern analyses into tables and plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_ROOT = REPO_ROOT / "experiments/cache_eval/results/vector_patterns"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="k100_seed42")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sift1m", "fashion_mnist", "glove_25", "lastfm_64"],
    )
    return parser.parse_args()


def load_summaries(args: argparse.Namespace) -> list[dict[str, object]]:
    summaries = []
    for dataset in args.datasets:
        path = args.input_root / dataset / args.run_name / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        summaries.append(json.loads(path.read_text()))
    return summaries


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(args)

    split_rows = []
    comparison_rows = []
    shared_reference_rows = []
    sequence_rows = []
    for summary in summaries:
        dataset = str(summary["dataset"])
        for split in summary["splits"]:
            split_rows.append({"dataset": dataset, **split})

        for comparison in summary["independent_center_comparisons"]:
            comparison_rows.append({"dataset": dataset, **comparison})

        for comparison in summary["shared_base_center_comparisons"]:
            shared_reference_rows.append(
                {
                    "dataset": dataset,
                    **{
                        key: value
                        for key, value in comparison.items()
                        if key != "other_counts_on_reference_centers"
                    },
                }
            )

        sequence = summary["query_sequence"]
        lag1 = next(
            (row for row in sequence["lag_same_cluster"] if row["lag"] == 1),
            None,
        )
        sequence_rows.append(
            {
                "dataset": dataset,
                "self_transition_probability": sequence[
                    "self_transition_probability"
                ],
                "run_length_mean": sequence["run_length_mean"],
                "transition_entropy_normalized": sequence[
                    "transition_entropy_normalized"
                ],
                "lag1_same_cluster_probability": lag1[
                    "same_cluster_probability"
                ]
                if lag1
                else None,
                "lag1_shuffled_probability": lag1["shuffled_probability"]
                if lag1
                else None,
                "lag1_locality_lift": lag1["locality_lift"] if lag1 else None,
            }
        )

    split_frame = pd.DataFrame(split_rows)
    comparison_frame = pd.DataFrame(comparison_rows)
    shared_reference_frame = pd.DataFrame(shared_reference_rows)
    sequence_frame = pd.DataFrame(sequence_rows)
    split_frame.to_csv(args.output_dir / "dataset_split_summary.csv", index=False)
    comparison_frame.to_csv(
        args.output_dir / "base_query_similarity_summary.csv", index=False
    )
    shared_reference_frame.to_csv(
        args.output_dir / "base_center_occupancy_summary.csv", index=False
    )
    sequence_frame.to_csv(
        args.output_dir / "query_sequence_locality_summary.csv", index=False
    )

    datasets = [str(summary["dataset"]) for summary in summaries]
    base_rows = split_frame[split_frame["name"] == "base"].set_index("dataset")
    query_rows = split_frame[split_frame["name"] == "query"].set_index("dataset")
    comparisons = comparison_frame.set_index("dataset")
    shared_references = shared_reference_frame.set_index("dataset")
    sequences = sequence_frame.set_index("dataset")

    x = np.arange(len(datasets))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(
        x - width / 2,
        [base_rows.loc[name, "cluster_size_gini"] for name in datasets],
        width,
        label="base",
    )
    axes[0, 0].bar(
        x + width / 2,
        [query_rows.loc[name, "cluster_size_gini"] for name in datasets],
        width,
        label="query",
    )
    axes[0, 0].set_title("Cluster-size Gini")
    axes[0, 0].legend()

    axes[0, 1].bar(
        x,
        [
            comparisons.loc[name, "matched_center_distance_over_radius_mean"]
            for name in datasets
        ],
    )
    axes[0, 1].set_title("Matched center distance / cluster radius")

    axes[1, 0].bar(
        x,
        [
            comparisons.loc[name, "matched_cluster_mass_js_divergence_bits"]
            for name in datasets
        ],
    )
    axes[1, 0].set_title("Matched base-query mass JS divergence")

    axes[1, 1].bar(
        x,
        [sequences.loc[name, "lag1_locality_lift"] for name in datasets],
    )
    axes[1, 1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Released query order: lag-1 locality lift")

    for axis in axes.flat:
        axis.set_xticks(x)
        axis.set_xticklabels(datasets, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "vector_pattern_overview.png", dpi=180)
    plt.close(figure)

    markdown = [
        "# Vector-pattern analysis summary",
        "",
        "## Query-order locality",
        "",
        "| dataset | self transition | lag-1 observed | lag-1 shuffled | lift | transition entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        row = sequences.loc[dataset]
        markdown.append(
            f"| {dataset} | {row['self_transition_probability']:.6f} | "
            f"{row['lag1_same_cluster_probability']:.6f} | "
            f"{row['lag1_shuffled_probability']:.6f} | "
            f"{row['lag1_locality_lift']:.4f} | "
            f"{row['transition_entropy_normalized']:.6f} |"
        )

    markdown.extend(
        [
            "",
            "## Base-query pattern similarity",
            "",
            "| dataset | center distance/radius | mass Spearman | mass JS divergence |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in datasets:
        row = comparisons.loc[dataset]
        markdown.append(
            f"| {dataset} | {row['matched_center_distance_over_radius_mean']:.6f} | "
            f"{row['matched_cluster_mass_spearman']} | "
            f"{row['matched_cluster_mass_js_divergence_bits']:.6f} |"
        )

    markdown.extend(
        [
            "",
            "## Query occupancy on base-fitted centers",
            "",
            "| dataset | occupancy Pearson | occupancy Spearman | occupancy JS divergence |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in datasets:
        row = shared_references.loc[dataset]
        markdown.append(
            f"| {dataset} | {row['occupancy_pearson']:.6f} | "
            f"{row['occupancy_spearman']:.6f} | "
            f"{row['occupancy_js_divergence_bits']:.6f} |"
        )

    (args.output_dir / "summary.md").write_text("\n".join(markdown) + "\n")
    print(f"wrote aggregate summary to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
