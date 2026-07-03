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
filter, thread count, cache capacities, result prefix, and log file. The search
binary writes result files using the recorded `result_prefix`, and stdout/stderr
is captured under `logs/`.

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
- admission reject percent when available;
- recall.

Current live-search output does not yet split static-hit and dynamic-hit rates
in the final summary. The trace path distinguishes cache sources in JSONL
records (`static`, `dynamic`, `disk`) when `DISKANN_CACHE_TRACE_PATH` is set, so
per-tier hit rates can be derived from traces until direct summary fields are
added.

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
