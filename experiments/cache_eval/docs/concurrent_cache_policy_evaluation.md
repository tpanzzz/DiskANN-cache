# Concurrent Cache Policy Evaluation

This note records the evaluation scaffold for the concurrent cache-policy work.
No SIFT1M benchmark results are committed by this branch; generated logs,
result binaries, and manifests should be written under ignored result
directories.

## Sweep Script

Run from the `DiskANN/` repository root:

```bash
experiments/cache_eval/scripts/run_concurrent_cache_policy_sweep.sh
```

The script defaults to:

- dataset directory: `experiments/cache_eval/sift1m`;
- index prefix: `experiments/cache_eval/sift1m/sift_disk`;
- result root: `experiments/cache_eval/sift1m/results/concurrent_cache`;
- workloads: `original`, `random_seed42`, `kmeans_k100_seed42`;
- threads: `1 2 4 8 16 32`;
- `search_list=100`;
- `beam_width=4`;
- `recall_at=10`;
- `search_io_limit=1000`;
- dynamic capacity: `10000`;
- static BFS tier nodes: `10000`;
- sharded dynamic cache shards: `64`.

Each run appends one row to:

```text
experiments/cache_eval/sift1m/results/concurrent_cache/run_manifest.csv
```

The manifest records workload, policy label, backend, shard count, admission
filter, thread count, cache capacities, query/truth files, search parameters,
result prefix, log file, and optional trace file. The search binary writes result
files using the recorded `result_prefix`, and stdout/stderr is captured under
`logs/`.

Set `TRACE_CACHE_ACCESS=1` to collect per-run cache-source JSONL traces next to
each result prefix. These traces are useful for splitting hits by static and
dynamic cache tier:

```bash
TRACE_CACHE_ACCESS=1 \
THREADS_LIST="1 8" \
WORKLOADS_LIST="original kmeans_k100_seed42" \
experiments/cache_eval/scripts/run_concurrent_cache_policy_sweep.sh
```

If a manifest already exists with an older header, the sweep exits and asks for a
fresh `RESULT_ROOT`. This avoids appending rows with incompatible schemas.

## Summary Script

After the sweep finishes, generate a consolidated CSV:

```bash
python3 experiments/cache_eval/scripts/summarize_concurrent_cache_results.py \
  --manifest experiments/cache_eval/sift1m/results/concurrent_cache/run_manifest.csv \
  --output experiments/cache_eval/sift1m/results/concurrent_cache/concurrent_cache_policy_summary.csv
```

Use `--dedupe-latest` if the manifest contains repeated rows for the same
`result_prefix` and only the last run should be summarized.

The summary script parses the fixed-width `search_disk_index` table from each
log. It also reads the query-file header to fill `queries` and computes
`total_ios_estimated = mean_ios_per_query * queries`. When a `trace_file` exists,
it adds `static_hit_percent`, `dynamic_hit_percent`, and `disk_miss_percent`
from `cache_source` JSONL records.

## Policies Covered

The default sweep covers:

- `NoCache`;
- immutable `StaticBFS`;
- global `CLOCK`;
- global `LRU`;
- sharded `CLOCK`;
- sharded `FIFO`;
- sharded `LRU`;
- sharded `CLOCK + TinyLFU admission`;
- `StaticBFS + sharded CLOCK`;
- `StaticBFS + sharded CLOCK + TinyLFU admission`.

The recommended first comparison is:

```bash
THREADS_LIST="1 8 16 32" \
WORKLOADS_LIST="original kmeans_k100_seed42" \
experiments/cache_eval/scripts/run_concurrent_cache_policy_sweep.sh
```

For a quick smoke run:

```bash
THREADS_LIST="1" \
WORKLOADS_LIST="original" \
SHARDS="4" \
experiments/cache_eval/scripts/run_concurrent_cache_policy_sweep.sh
```

## Manual Command Examples

Sharded CLOCK:

```bash
cargo run -p diskann-tools --bin search_disk_index --release -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix experiments/cache_eval/sift1m/sift_disk \
  --result_output_prefix experiments/cache_eval/sift1m/results/concurrent_cache/original_sharded_clock_8t \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --ground_truth_file experiments/cache_eval/sift1m/groundtruth.bin \
  --search_list 100 \
  --beam_width 4 \
  --recall_at 10 \
  --search_io_limit 1000 \
  --num_threads 8 \
  --cache_policy clock \
  --cache_capacity 10000 \
  --cache_backend sharded \
  --cache_shards 64
```

StaticBFS plus sharded CLOCK:

```bash
cargo run -p diskann-tools --bin search_disk_index --release -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix experiments/cache_eval/sift1m/sift_disk \
  --result_output_prefix experiments/cache_eval/sift1m/results/concurrent_cache/original_static_bfs_sharded_clock_8t \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --ground_truth_file experiments/cache_eval/sift1m/groundtruth.bin \
  --search_list 100 \
  --beam_width 4 \
  --recall_at 10 \
  --search_io_limit 1000 \
  --num_threads 8 \
  --cache_policy clock \
  --cache_capacity 10000 \
  --cache_backend sharded \
  --cache_shards 64 \
  --cache_static_nodes 10000
```

StaticBFS plus sharded CLOCK with TinyLFU admission:

```bash
cargo run -p diskann-tools --bin search_disk_index --release -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix experiments/cache_eval/sift1m/sift_disk \
  --result_output_prefix experiments/cache_eval/sift1m/results/concurrent_cache/original_static_bfs_sharded_clock_tiny_lfu_8t \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --ground_truth_file experiments/cache_eval/sift1m/groundtruth.bin \
  --search_list 100 \
  --beam_width 4 \
  --recall_at 10 \
  --search_io_limit 1000 \
  --num_threads 8 \
  --cache_policy clock \
  --cache_capacity 10000 \
  --cache_backend sharded \
  --cache_shards 64 \
  --cache_admission tiny_lfu \
  --cache_static_nodes 10000
```

## Result Summary Fields

When summarizing logs and result outputs, use at least:

- workload;
- policy;
- backend;
- shards;
- admission;
- threads;
- cache capacity nodes;
- static cache nodes;
- QPS;
- mean latency;
- p999 latency;
- mean IO/query;
- cache hit percent;
- static hit percent when trace is available;
- dynamic hit percent when trace is available;
- disk miss percent when trace is available;
- admission reject percent when available;
- recall;
- QPS speedups versus `NoCache`, `StaticBFS`, and `global_clock` baselines;
- mean-IO deltas versus `NoCache` and `global_clock` baselines;
- cache-hit delta versus `global_clock`.

Current live-search output does not yet print admission rejection counts or
direct static/dynamic tier summary fields. The summary script leaves unavailable
fields blank and derives per-tier rates from traces when `TRACE_CACHE_ACCESS=1`
was used for the sweep.

## Acceptance Checks

Use the criteria from
`experiments/cache_eval/docs/cache_policy_concurrency_strategy_plan.md`:

- recall remains consistent with the existing SIFT1M runs;
- sharded CLOCK is close to global CLOCK in one-thread runs;
- sharded CLOCK improves QPS over global CLOCK/LRU at 8+ threads;
- `kmeans_k100_seed42` sharded CLOCK remains near the current 36-37% hit-rate
  range unless shard-local capacity explains a difference;
- StaticBFS plus sharded CLOCK is not materially slower than immutable StaticBFS
  on original/random workloads;
- admission stays opt-in unless it improves QPS or IO.
