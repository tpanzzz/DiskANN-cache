#!/usr/bin/env python3
"""Summarize static-cache source benchmark logs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


FIELDS = [
    "dataset", "source", "repetition", "cache_capacity_nodes", "queries",
    "search_list", "beam_width", "qps", "mean_latency_us", "p999_latency_us",
    "mean_ios_per_query", "cache_hit_percent", "recall_percent", "log_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_metric_line(log_file: Path) -> dict[str, float]:
    candidates = []
    for line in log_file.read_text(errors="replace").splitlines():
        tokens = line.strip().split()
        if len(tokens) != 13 or not tokens[0].isdigit():
            continue
        try:
            values = [float(token) for token in tokens]
        except ValueError:
            continue
        candidates.append(values)
    if len(candidates) != 1:
        raise ValueError(f"expected one metric row in {log_file}, found {len(candidates)}")
    row = candidates[0]
    return {
        "search_list": int(row[0]),
        "beam_width": int(row[1]),
        "qps": row[2],
        "mean_latency_us": row[3],
        "p999_latency_us": row[4],
        "mean_ios_per_query": row[5],
        "cache_hit_percent": row[11],
        "recall_percent": row[12],
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="") as file:
        runs = list(csv.DictReader(file))
    detailed = []
    for run in runs:
        metrics = parse_metric_line(Path(run["log_file"]))
        detailed.append({**run, **metrics})
    write_csv(args.output_dir / "benchmark_repetitions.csv", detailed, FIELDS)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in detailed:
        grouped.setdefault((row["dataset"], row["source"]), []).append(row)
    summary = []
    metric_names = [
        "qps", "mean_latency_us", "p999_latency_us", "mean_ios_per_query",
        "cache_hit_percent", "recall_percent",
    ]
    for (dataset, source), rows in sorted(grouped.items()):
        output = {
            "dataset": dataset,
            "source": source,
            "repetitions": len(rows),
            "cache_capacity_nodes": rows[0]["cache_capacity_nodes"],
            "queries": rows[0]["queries"],
            "search_list": rows[0]["search_list"],
            "beam_width": rows[0]["beam_width"],
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            output[f"median_{metric}"] = statistics.median(values)
            output[f"min_{metric}"] = min(values)
            output[f"max_{metric}"] = max(values)
        summary.append(output)
    summary_fields = list(summary[0])
    write_csv(args.output_dir / "benchmark_summary.csv", summary, summary_fields)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote benchmark summary to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
