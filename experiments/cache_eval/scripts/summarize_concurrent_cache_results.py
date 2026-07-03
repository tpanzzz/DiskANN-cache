#!/usr/bin/env python3
"""Summarize concurrent cache-policy search logs into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

OUTPUT_FIELDS = [
    "workload",
    "policy",
    "backend",
    "shards",
    "admission",
    "threads",
    "cache_capacity_nodes",
    "static_cache_nodes",
    "queries",
    "search_list",
    "beam_width",
    "recall_at",
    "search_io_limit",
    "qps",
    "mean_latency_us",
    "p999_latency_us",
    "mean_ios_per_query",
    "total_ios_estimated",
    "mean_io_us",
    "cpu_us",
    "pq_preprocess_us",
    "mean_comps",
    "mean_hops",
    "cache_hit_percent",
    "static_hit_percent",
    "dynamic_hit_percent",
    "disk_miss_percent",
    "admission_reject_percent",
    "recall_percent",
    "qps_speedup_vs_nocache",
    "qps_speedup_vs_static_bfs",
    "qps_speedup_vs_global_clock",
    "mean_ios_delta_vs_nocache",
    "mean_ios_delta_vs_global_clock",
    "cache_hit_delta_vs_global_clock",
    "result_prefix",
    "log_file",
    "trace_file",
    "status",
]

SEARCH_PARAMS_RE = re.compile(
    r"#threads:\s*(?P<threads>\d+),\s*"
    r"recall_at\s*(?P<recall_at>\d+),\s*"
    r"search_list_size:\s*\[(?P<search_list>[^\]]*)\],\s*"
    r"search_io_limit:\s*(?P<search_io_limit>\d+),.*"
    r"beam_width:\s*(?P<beam_width>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize concurrent cache-policy evaluation logs."
    )
    parser.add_argument(
        "--manifest",
        default="experiments/cache_eval/sift1m/results/concurrent_cache/run_manifest.csv",
        help="Run manifest written by run_concurrent_cache_policy_sweep.sh.",
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/cache_eval/sift1m/results/concurrent_cache/"
            "concurrent_cache_policy_summary.csv"
        ),
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--dedupe-latest",
        action="store_true",
        help="Keep only the latest manifest row per result_prefix before summarizing.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a parser self-test using temporary files and exit.",
    )
    return parser.parse_args()


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def normalize_float(value: str | float | int | None, digits: int = 2) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def normalize_int(value: str | float | int | None) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_search_log(log_file: Path) -> tuple[list[dict[str, str]], dict[str, str], str]:
    if not str(log_file) or not log_file.is_file():
        return [], {}, "missing_log"

    params: dict[str, str] = {}
    metrics: list[dict[str, str]] = []

    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = SEARCH_PARAMS_RE.search(line)
            if match:
                params.update(match.groupdict())

            tokens = line.strip().split()
            if len(tokens) not in (12, 13):
                continue
            if not tokens[0].isdigit() or not all(is_float(token) for token in tokens[1:]):
                continue

            row = {
                "search_list": normalize_int(tokens[0]),
                "beam_width": normalize_int(tokens[1]),
                "qps": normalize_float(tokens[2]),
                "mean_latency_us": normalize_float(tokens[3]),
                "p999_latency_us": normalize_float(tokens[4]),
                "mean_ios_per_query": normalize_float(tokens[5]),
                "mean_io_us": normalize_float(tokens[6]),
                "cpu_us": normalize_float(tokens[7]),
                "pq_preprocess_us": normalize_float(tokens[8]),
                "mean_comps": normalize_float(tokens[9]),
                "mean_hops": normalize_float(tokens[10]),
                "cache_hit_percent": normalize_float(tokens[11]),
                "recall_percent": normalize_float(tokens[12]) if len(tokens) == 13 else "",
            }
            metrics.append(row)

    return metrics, params, "ok" if metrics else "no_metrics"


def read_query_count(query_file: str | None) -> str:
    if not query_file:
        return ""
    path = Path(query_file)
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            header = handle.read(8)
        if len(header) != 8:
            return ""
        nrows, _dim = struct.unpack("<II", header)
        return str(nrows)
    except OSError:
        return ""


def parse_trace(trace_file: str | None) -> dict[str, str]:
    empty = {
        "static_hit_percent": "",
        "dynamic_hit_percent": "",
        "disk_miss_percent": "",
    }
    if not trace_file:
        return empty
    path = Path(trace_file)
    if not path.exists():
        return empty

    counts: defaultdict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("stage") != "load_vertices":
                continue
            counts["total"] += 1
            source = str(event.get("cache_source", ""))
            if source == "static":
                counts["static"] += 1
            elif source == "dynamic":
                counts["dynamic"] += 1
            elif source == "disk":
                counts["disk"] += 1

    total = counts["total"]
    if total == 0:
        return empty
    return {
        "static_hit_percent": normalize_float(100.0 * counts["static"] / total),
        "dynamic_hit_percent": normalize_float(100.0 * counts["dynamic"] / total),
        "disk_miss_percent": normalize_float(100.0 * counts["disk"] / total),
    }


def maybe_total_ios(mean_ios: str, queries: str) -> str:
    mean = optional_float(mean_ios)
    count = optional_float(queries)
    if mean is None or count is None:
        return ""
    return str(int(round(mean * count)))


def load_manifest(path: Path, dedupe_latest: bool) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not dedupe_latest:
        return rows

    latest: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        key = row.get("result_prefix") or f"__row_{index}"
        latest[key] = row
    return list(latest.values())


def first_search_list_value(value: str | None) -> str:
    if not value:
        return ""
    return value.split(",")[0].strip()


def build_rows(manifest_rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for manifest_row in manifest_rows:
        log_file = Path(manifest_row.get("log_file") or "")
        metric_rows, params, status = parse_search_log(log_file)
        if not metric_rows:
            metric_rows = [{}]

        query_count = manifest_row.get("queries") or read_query_count(manifest_row.get("query_file"))
        trace_stats = parse_trace(manifest_row.get("trace_file"))

        for metric_row in metric_rows:
            search_list = (
                metric_row.get("search_list")
                or manifest_row.get("search_list")
                or first_search_list_value(params.get("search_list"))
            )
            beam_width = (
                metric_row.get("beam_width")
                or manifest_row.get("beam_width")
                or params.get("beam_width")
            )
            recall_at = manifest_row.get("recall_at") or params.get("recall_at")
            search_io_limit = manifest_row.get("search_io_limit") or params.get("search_io_limit")

            row = {field: "" for field in OUTPUT_FIELDS}
            row.update(
                {
                    "workload": manifest_row.get("workload", ""),
                    "policy": manifest_row.get("policy", ""),
                    "backend": manifest_row.get("backend", ""),
                    "shards": manifest_row.get("shards", ""),
                    "admission": manifest_row.get("admission", ""),
                    "threads": manifest_row.get("threads") or params.get("threads", ""),
                    "cache_capacity_nodes": manifest_row.get("cache_capacity_nodes", ""),
                    "static_cache_nodes": manifest_row.get("static_cache_nodes", ""),
                    "queries": query_count,
                    "search_list": normalize_int(search_list),
                    "beam_width": normalize_int(beam_width),
                    "recall_at": normalize_int(recall_at),
                    "search_io_limit": normalize_int(search_io_limit),
                    "result_prefix": manifest_row.get("result_prefix", ""),
                    "log_file": manifest_row.get("log_file", ""),
                    "trace_file": manifest_row.get("trace_file", ""),
                    "status": status,
                }
            )
            row.update(metric_row)
            row.update(trace_stats)
            row["total_ios_estimated"] = maybe_total_ios(
                row["mean_ios_per_query"], row["queries"]
            )
            output_rows.append(row)

    add_baseline_deltas(output_rows)
    return output_rows


def add_baseline_deltas(rows: list[dict[str, str]]) -> None:
    groups: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["workload"], row["threads"], row["search_list"])
        groups[key][row["policy"].lower()] = row

    for row in rows:
        key = (row["workload"], row["threads"], row["search_list"])
        baselines = groups[key]
        qps = optional_float(row.get("qps"))
        mean_ios = optional_float(row.get("mean_ios_per_query"))
        cache_hit = optional_float(row.get("cache_hit_percent"))

        set_speedup(row, "qps_speedup_vs_nocache", qps, baselines.get("nocache"))
        set_speedup(row, "qps_speedup_vs_static_bfs", qps, baselines.get("staticbfs"))
        set_speedup(row, "qps_speedup_vs_global_clock", qps, baselines.get("global_clock"))
        set_delta(
            row,
            "mean_ios_delta_vs_nocache",
            mean_ios,
            baselines.get("nocache"),
            "mean_ios_per_query",
        )
        set_delta(
            row,
            "mean_ios_delta_vs_global_clock",
            mean_ios,
            baselines.get("global_clock"),
            "mean_ios_per_query",
        )
        set_delta(
            row,
            "cache_hit_delta_vs_global_clock",
            cache_hit,
            baselines.get("global_clock"),
            "cache_hit_percent",
        )


def set_speedup(
    row: dict[str, str], field: str, value: float | None, baseline: dict[str, str] | None
) -> None:
    if value is None or not baseline:
        return
    base = optional_float(baseline.get("qps"))
    if base is None or base == 0.0:
        return
    row[field] = normalize_float(value / base, digits=4)


def set_delta(
    row: dict[str, str],
    field: str,
    value: float | None,
    baseline: dict[str, str] | None,
    baseline_field: str,
) -> None:
    if value is None or not baseline:
        return
    base = optional_float(baseline.get(baseline_field))
    if base is None:
        return
    row[field] = normalize_float(value - base, digits=4)


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        query = root / "query.fbin"
        query.write_bytes(struct.pack("<II", 10_000, 128))

        nocache_log = root / "nocache.log"
        nocache_log.write_text(
            "Search parameters: #threads: 8, recall_at 10, search_list_size: [100], "
            "search_io_limit: 1000, fail_if_recall_below: 0, beam_width: 4\n"
            "100   4.00        320.00         3100.00             4200.00             "
            "114.08      2800.00         300.00    3.50                4150.00     "
            "114.00      0.00          97.04           \n",
            encoding="utf-8",
        )

        clock_log = root / "clock.log"
        clock_log.write_text(
            "Search parameters: #threads: 8, recall_at 10, search_list_size: [100], "
            "search_io_limit: 1000, fail_if_recall_below: 0, beam_width: 4\n"
            "L     Beamwidth   QPS            Mean Latency (us)   99.9 Latency (us) "
            "Mean IOs    Mean IO (us)    CPU (us)  PQ Preprocess (us)Mean Comps  "
            "Mean Hops   Cache Hit %  Recall@10       \n"
            "100   4.00        383.62         2605.45             3960.00             "
            "71.89       2000.10         300.20    3.50                4150.00     "
            "114.00      36.98         97.04           \n",
            encoding="utf-8",
        )

        trace = root / "trace.jsonl"
        trace.write_text(
            '{"stage":"load_vertices","vertex_id":1,"cache_hit":true,"cache_source":"static"}\n'
            '{"stage":"load_vertices","vertex_id":2,"cache_hit":true,"cache_source":"static"}\n'
            '{"stage":"load_vertices","vertex_id":3,"cache_hit":true,"cache_source":"dynamic"}\n'
            '{"stage":"load_vertices","vertex_id":4,"cache_hit":false,"cache_source":"disk"}\n',
            encoding="utf-8",
        )

        manifest = root / "run_manifest.csv"
        manifest.write_text(
            "workload,policy,backend,shards,admission,threads,cache_capacity_nodes,"
            "static_cache_nodes,query_file,ground_truth_file,search_list,beam_width,"
            "recall_at,search_io_limit,result_prefix,log_file,trace_file\n"
            f"original,NoCache,none,0,none,8,0,0,{query},,100,4,10,"
            f"1000,{root / 'nocache_result'},{nocache_log},\n"
            f"original,sharded_clock,sharded,64,none,8,10000,0,{query},,100,4,10,"
            f"1000,{root / 'result'},{clock_log},{trace}\n",
            encoding="utf-8",
        )

        rows = build_rows(load_manifest(manifest, dedupe_latest=False))
        assert len(rows) == 2, rows
        row = rows[1]
        assert row["queries"] == "10000", row
        assert row["qps"] == "383.62", row
        assert row["total_ios_estimated"] == "718900", row
        assert row["static_hit_percent"] == "50.00", row
        assert row["dynamic_hit_percent"] == "25.00", row
        assert row["disk_miss_percent"] == "25.00", row
        assert row["qps_speedup_vs_nocache"] == "1.1988", row
        assert row["mean_ios_delta_vs_nocache"] == "-42.1900", row
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"manifest not found: {manifest}", file=sys.stderr)
        return 1

    rows = build_rows(load_manifest(manifest, args.dedupe_latest))
    write_summary(rows, Path(args.output))
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
