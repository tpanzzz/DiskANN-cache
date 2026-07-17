#!/usr/bin/env python3
"""Aggregate node-expansion summaries from the spatial-locality pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def identify_run(pilot_root: Path, summary_path: Path) -> tuple[str, str]:
    relative = summary_path.relative_to(pilot_root)
    dataset = relative.parts[0]
    expansions_index = relative.parts.index("expansions")
    run_parts = relative.parts[expansions_index + 1 : -2]
    return dataset, "/".join(run_parts) if run_parts else "default"


def collect_summaries(pilot_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(pilot_root.glob("*/expansions/**/summary/summary.json")):
        summary = json.loads(path.read_text())
        dataset, run_tag = identify_run(pilot_root, path)
        num_nodes = int(summary["num_nodes"])
        rows.append(
            {
                "dataset": dataset,
                "run_tag": run_tag,
                "num_nodes": num_nodes,
                "total_expansions": int(summary["total_expansions"]),
                "unique_expanded_nodes": int(summary["unique_expanded_nodes"]),
                "expanded_node_fraction": float(summary["expanded_node_fraction"]),
                "gini": float(summary["gini"]),
                "top_1_percent_nodes_expansion_fraction": float(
                    summary["top_1_percent_nodes_expansion_fraction"]
                ),
                "top_5_percent_nodes_expansion_fraction": float(
                    summary["top_5_percent_nodes_expansion_fraction"]
                ),
                "nodes_for_50_percent_expansions": int(
                    summary["nodes_for_50_percent_expansions"]
                ),
                "nodes_for_90_percent_expansions": int(
                    summary["nodes_for_90_percent_expansions"]
                ),
                "nodes_for_95_percent_expansions": int(
                    summary["nodes_for_95_percent_expansions"]
                ),
                "malformed_trace_lines": int(summary["malformed_trace_lines"]),
                "summary_path": str(path.resolve()),
            }
        )
    if not rows:
        raise ValueError(f"no expansion summaries found below {pilot_root}")

    frame = pd.DataFrame(rows)
    for mass in (50, 90, 95):
        frame[f"node_fraction_for_{mass}_percent_expansions"] = (
            frame[f"nodes_for_{mass}_percent_expansions"] / frame["num_nodes"]
        )
    return frame


def save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    labels = [
        dataset if tag == "default" else f"{dataset}\n{tag}"
        for dataset, tag in zip(frame["dataset"], frame["run_tag"])
    ]
    positions = np.arange(len(frame))

    figure, axis = plt.subplots(figsize=(max(8, 1.6 * len(frame)), 5))
    width = 0.36
    axis.bar(
        positions - width / 2,
        frame["top_1_percent_nodes_expansion_fraction"],
        width,
        label="Top 1% of nodes",
    )
    axis.bar(
        positions + width / 2,
        frame["top_5_percent_nodes_expansion_fraction"],
        width,
        label="Top 5% of nodes",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Fraction of all node expansions")
    axis.set_ylim(0, 1)
    axis.set_title("Expansion mass concentrated in frequently visited nodes")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "expansion_top_node_share.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(max(8, 1.6 * len(frame)), 5))
    width = 0.24
    for offset, mass in zip((-width, 0, width), (50, 90, 95)):
        axis.bar(
            positions + offset,
            frame[f"node_fraction_for_{mass}_percent_expansions"],
            width,
            label=f"{mass}% of expansions",
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Fraction of graph nodes required")
    axis.set_ylim(0, 1)
    axis.set_title("Graph footprint required to cover expansion mass")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "expansion_mass_coverage.png", dpi=180)
    plt.close(figure)


def save_markdown(frame: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Spatial-locality pilot: node-expansion summary",
        "",
        "| Dataset | Run | Nodes | Expansions | Expanded nodes | Gini | Top 1% share | Top 5% share | Nodes for 50% | Nodes for 90% | Malformed lines |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.run_tag} | {row.num_nodes:,} | "
            f"{row.total_expansions:,} | {row.expanded_node_fraction:.2%} | "
            f"{row.gini:.4f} | "
            f"{row.top_1_percent_nodes_expansion_fraction:.2%} | "
            f"{row.top_5_percent_nodes_expansion_fraction:.2%} | "
            f"{row.node_fraction_for_50_percent_expansions:.2%} | "
            f"{row.node_fraction_for_90_percent_expansions:.2%} | "
            f"{row.malformed_trace_lines} |"
        )
    lines.extend(
        [
            "",
            "Generated plots:",
            "",
            "- `expansion_top_node_share.png`",
            "- `expansion_mass_coverage.png`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = collect_summaries(args.pilot_root)
    frame.to_csv(args.output_dir / "node_expansion_summary.csv", index=False)
    save_plots(frame, args.output_dir)
    save_markdown(frame, args.output_dir)
    print(f"wrote {len(frame)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
