#!/usr/bin/env python3
"""Summarize the GloVe PQ, normalization, and full-precision sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recall-threshold", type=float, default=95.0)
    return parser.parse_args()


def variant_label(row: pd.Series) -> str:
    prefix = "normalized" if row["representation"] == "normalized" else "raw"
    if row["navigation"] == "full_precision":
        return f"{prefix} FP"
    return f"{prefix} PQ{int(row['pq_chunks'])}"


def select_threshold_rows(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    qualifying = frame[frame["recall_at_10_percent"] >= threshold].copy()
    if qualifying.empty:
        raise ValueError(f"no row reached Recall@10 >= {threshold}")
    selected = (
        qualifying.sort_values("search_l")
        .groupby(["representation", "navigation", "pq_chunks"], as_index=False)
        .first()
    )
    selected["variant"] = selected.apply(variant_label, axis=1)
    selected["navigation_mib"] = selected["navigation_bytes"] / (1024 * 1024)
    selected["max_rss_mib"] = selected["max_rss_kib"] / 1024
    return selected


def save_recall_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for keys, rows in frame.groupby(["representation", "navigation", "pq_chunks"]):
        rows = rows.sort_values("search_l")
        label = variant_label(rows.iloc[0])
        axis.plot(
            rows["search_l"],
            rows["recall_at_10_percent"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=label,
        )
    axis.axhline(95, color="black", linestyle="--", linewidth=1, label="95% target")
    axis.set_xscale("log")
    axis.set_xlabel("Search list L")
    axis.set_ylabel("Recall@10 (%)")
    axis.set_ylim(20, 100.5)
    axis.set_title("GloVe-25 recall: PQ granularity, normalization, and exact navigation")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "recall_vs_search_l.png", dpi=180)
    plt.close(figure)


def save_threshold_plot(selected: pd.DataFrame, output_dir: Path) -> None:
    order = [
        "raw PQ5",
        "raw PQ10",
        "raw PQ16",
        "raw PQ25",
        "raw FP",
        "normalized PQ5",
        "normalized PQ10",
        "normalized PQ16",
        "normalized PQ25",
        "normalized FP",
    ]
    selected = selected.set_index("variant").loc[order].reset_index()
    positions = np.arange(len(selected))

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    axes[0].bar(positions, selected["search_l"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Minimum tested L meeting 95% recall")
    axes[0].set_title("Search budget")

    axes[1].bar(positions, selected["mean_ios"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean I/O operations per query")
    axes[1].set_title("Disk work at target recall")

    axes[2].bar(positions, selected["navigation_mib"])
    axes[2].set_ylabel("Navigation-vector memory (MiB)")
    axes[2].set_title("PQ codes or full vectors")

    for axis in axes:
        axis.set_xticks(positions, selected["variant"], rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "target_recall_tradeoffs.png", dpi=180)
    plt.close(figure)


def save_markdown(selected: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    selected = selected.sort_values(
        ["representation", "navigation", "pq_chunks"],
        ascending=[False, True, True],
    )
    lines = [
        "# GloVe PQ and full-precision sweep summary",
        "",
        f"Minimum tested L reaching Recall@10 >= {threshold:.2f}%:",
        "",
        "| Variant | L | Recall@10 | QPS | Mean latency (us) | Mean I/O | Navigation MiB | Max RSS MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.search_l} | {row.recall_at_10_percent:.2f}% | "
            f"{row.qps:.2f} | {row.mean_latency_us:.2f} | {row.mean_ios:.2f} | "
            f"{row.navigation_mib:.2f} | {row.max_rss_mib:.2f} |"
        )
    lines.extend(
        [
            "",
            "Generated plots:",
            "",
            "- `recall_vs_search_l.png`",
            "- `target_recall_tradeoffs.png`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    frame = frame.drop_duplicates(
        ["representation", "navigation", "pq_chunks", "search_l"], keep="last"
    )
    frame["variant"] = frame.apply(variant_label, axis=1)
    selected = select_threshold_rows(frame, args.recall_threshold)
    selected.to_csv(args.output_dir / "selected_target_recall.csv", index=False)
    save_recall_plot(frame, args.output_dir)
    save_threshold_plot(selected, args.output_dir)
    save_markdown(selected, args.output_dir, args.recall_threshold)
    print(f"wrote GloVe sweep summary to {args.output_dir}")


if __name__ == "__main__":
    main()
