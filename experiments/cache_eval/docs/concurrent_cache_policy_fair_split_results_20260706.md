# Fair Hybrid Split Sweep Results, 2026-07-06

This document summarizes the equal-capacity SIFT1M live-search sweep for
hybrid `StaticBFS + sharded CLOCK` cache policies on branch
`eval/concurrent-cache-policies`.

## Purpose

The previous full concurrent-cache sweep used hybrid policies with
`static_cache_nodes=10000` plus `cache_capacity_nodes=10000`. That made the
hybrid policies 20,000-node caches, while `StaticBFS`, `global_clock`, and
other single-tier baselines used 10,000 nodes. Those results are useful as an
"extra dynamic tier" experiment, but not as an equal-capacity comparison.

This run fixes that by enforcing:

```text
static_cache_nodes + dynamic_cache_nodes = 10000
```

for every hybrid policy. The 50/50 hybrid therefore means:

```text
--cache_static_nodes 5000 --cache_capacity 5000
```

not 10,000 static plus 10,000 dynamic.

## Run Configuration

The sweep was run from the `DiskANN/` repository root:

```bash
RESULT_ROOT=experiments/cache_eval/sift1m/results/concurrent_cache_fair_split_20260706 \
TOTAL_CACHE_NODES=10000 \
SPLITS="2500:7500 5000:5000 7500:2500" \
THREADS_LIST="1 2 4 8 16 32" \
WORKLOADS_LIST="original random_seed42 kmeans_k100_seed42" \
experiments/cache_eval/scripts/run_fair_hybrid_split_sweep.sh
```

The consolidated summary was generated with:

```bash
python3 experiments/cache_eval/scripts/summarize_concurrent_cache_results.py \
  --manifest experiments/cache_eval/sift1m/results/concurrent_cache_fair_split_20260706/run_manifest.csv \
  --output experiments/cache_eval/sift1m/results/concurrent_cache_fair_split_20260706/concurrent_cache_policy_summary.csv \
  --dedupe-latest
```

Dataset and search settings:

- dataset: SIFT1M;
- index prefix: `experiments/cache_eval/sift1m/sift_disk`;
- workloads: `original`, `random_seed42`, `kmeans_k100_seed42`;
- threads: `1, 2, 4, 8, 16, 32`;
- `search_list=100`, `beam_width=4`, `recall_at=10`,
  `search_io_limit=1000`;
- sharded cache: 64 shards;
- `TRACE_CACHE_ACCESS=0`, so the run records total cache hit percentage but
  not per-tier static/dynamic hit split.

## Capacity Matrix

| policy family | static nodes | dynamic nodes | total nodes |
| --- | ---: | ---: | ---: |
| `NoCache` | 0 | 0 | 0 |
| `StaticBFS` | 10000 | 0 | 10000 |
| `global_clock`, `global_lru` | 0 | 10000 | 10000 |
| `sharded_clock`, `sharded_fifo`, `sharded_lru` | 0 | 10000 | 10000 |
| `sharded_clock_tiny_lfu` | 0 | 10000 | 10000 |
| `static_bfs_sharded_clock_s2500_d7500` | 2500 | 7500 | 10000 |
| `static_bfs_sharded_clock_s5000_d5000` | 5000 | 5000 | 10000 |
| `static_bfs_sharded_clock_s7500_d2500` | 7500 | 2500 | 10000 |
| `static_bfs_sharded_clock_tiny_lfu_s2500_d7500` | 2500 | 7500 | 10000 |
| `static_bfs_sharded_clock_tiny_lfu_s5000_d5000` | 5000 | 5000 | 10000 |
| `static_bfs_sharded_clock_tiny_lfu_s7500_d2500` | 7500 | 2500 | 10000 |

Each workload/thread group has 14 policies: 8 single-tier or non-hybrid
baselines plus 3 non-admission hybrid splits and 3 TinyLFU hybrid splits.

## Output Files

Primary files:

- manifest:
  `experiments/cache_eval/sift1m/results/concurrent_cache_fair_split_20260706/run_manifest.csv`;
- consolidated summary:
  `experiments/cache_eval/sift1m/results/concurrent_cache_fair_split_20260706/concurrent_cache_policy_summary.csv`;
- per-run logs:
  `experiments/cache_eval/sift1m/results/concurrent_cache_fair_split_20260706/logs/`.

Derived files:

- `completeness_report.txt`;
- `best_policy_by_workload_thread.csv`;
- `split_comparison_by_workload_thread.csv`;
- `best_split_by_workload_thread.csv`;
- `policy_aggregate_by_workload.csv`;
- `fair_50_50_key_comparisons.csv`;
- `markdown_32t_snapshot.md`.

## Completeness Checks

| check | result |
| --- | ---: |
| expected rows | 252 |
| summarized rows | 252 |
| status `ok` rows | 252 |
| failed rows | 0 |
| workload/thread groups | 18 |
| policies per workload/thread group | 14 |
| Recall@10 range | 97.04% to 97.04% |

Recall stayed identical across all cache variants. The observed performance
differences are therefore cache-path and I/O differences, not search-quality
differences.

## Fair 50/50 Hybrid At 32 Threads

This table compares the equal-capacity 50/50 hybrid against key baselines at
32 threads. Positive QPS delta is better; negative I/O delta is better.

| workload | comparison | target QPS | baseline QPS | QPS delta | I/O delta | hit delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original | 50/50 CLOCK vs StaticBFS | 4465.62 | 4479.03 | -0.30% | -0.61 | +0.53 pp |
| original | 50/50 CLOCK vs sharded CLOCK | 4465.62 | 4407.95 | +1.31% | -1.32 | +1.16 pp |
| original | 50/50 TinyLFU vs 50/50 CLOCK | 4521.13 | 4465.62 | +1.24% | -0.84 | +0.74 pp |
| random_seed42 | 50/50 CLOCK vs StaticBFS | 4466.57 | 4490.10 | -0.52% | -0.51 | +0.44 pp |
| random_seed42 | 50/50 CLOCK vs sharded CLOCK | 4466.57 | 4390.73 | +1.73% | -1.54 | +1.35 pp |
| random_seed42 | 50/50 TinyLFU vs 50/50 CLOCK | 4529.21 | 4466.57 | +1.40% | -0.78 | +0.69 pp |
| kmeans_k100_seed42 | 50/50 CLOCK vs StaticBFS | 4793.22 | 4494.86 | +6.64% | -9.20 | +8.06 pp |
| kmeans_k100_seed42 | 50/50 CLOCK vs sharded CLOCK | 4793.22 | 4844.95 | -1.07% | +1.29 | -1.13 pp |
| kmeans_k100_seed42 | 50/50 TinyLFU vs 50/50 CLOCK | 4679.14 | 4793.22 | -2.38% | +3.69 | -3.24 pp |

Key point: the fair 50/50 hybrid no longer strongly dominates the single-tier
baselines. It is close to `StaticBFS` on `original` and `random_seed42`, better
than plain `sharded_clock` on those workloads, but slightly worse than
`sharded_clock` on the locality-heavy `kmeans_k100_seed42` workload.

## 32-Thread Snapshot

### `original`

| policy | QPS | mean I/O/query | hit % | p99.9 us |
| --- | ---: | ---: | ---: | ---: |
| `NoCache` | 3936.78 | 114.08 | 0.00 | 12311 |
| `StaticBFS` | 4479.03 | 104.32 | 8.56 | 11260 |
| `sharded_clock` | 4407.95 | 105.03 | 7.93 | 9398 |
| `sharded_clock_tiny_lfu` | 4508.19 | 102.70 | 9.97 | 13291 |
| `static_bfs_sharded_clock_s2500_d7500` | 4475.53 | 103.55 | 9.23 | 9144 |
| `static_bfs_sharded_clock_s5000_d5000` | 4465.62 | 103.71 | 9.09 | 11606 |
| `static_bfs_sharded_clock_s7500_d2500` | 4483.88 | 104.08 | 8.76 | 9057 |
| `static_bfs_sharded_clock_tiny_lfu_s5000_d5000` | 4521.13 | 102.87 | 9.83 | 12822 |

### `random_seed42`

| policy | QPS | mean I/O/query | hit % | p99.9 us |
| --- | ---: | ---: | ---: | ---: |
| `NoCache` | 3941.66 | 114.08 | 0.00 | 11783 |
| `StaticBFS` | 4490.10 | 104.32 | 8.56 | 10435 |
| `sharded_clock` | 4390.73 | 105.35 | 7.65 | 10526 |
| `sharded_clock_tiny_lfu` | 4532.94 | 102.84 | 9.85 | 8986 |
| `static_bfs_sharded_clock_s2500_d7500` | 4463.81 | 103.75 | 9.06 | 11498 |
| `static_bfs_sharded_clock_s5000_d5000` | 4466.57 | 103.81 | 9.00 | 11273 |
| `static_bfs_sharded_clock_s7500_d2500` | 4465.15 | 104.12 | 8.73 | 10640 |
| `static_bfs_sharded_clock_tiny_lfu_s5000_d5000` | 4529.21 | 103.03 | 9.69 | 8949 |

### `kmeans_k100_seed42`

| policy | QPS | mean I/O/query | hit % | p99.9 us |
| --- | ---: | ---: | ---: | ---: |
| `NoCache` | 3946.46 | 114.08 | 0.00 | 11481 |
| `StaticBFS` | 4494.86 | 104.32 | 8.56 | 10420 |
| `sharded_clock` | 4844.95 | 93.83 | 17.75 | 8769 |
| `sharded_clock_tiny_lfu` | 4692.32 | 97.82 | 14.25 | 10934 |
| `static_bfs_sharded_clock_s2500_d7500` | 4863.65 | 93.40 | 18.13 | 8654 |
| `static_bfs_sharded_clock_s5000_d5000` | 4793.22 | 95.12 | 16.62 | 12063 |
| `static_bfs_sharded_clock_s7500_d2500` | 4539.92 | 100.91 | 11.54 | 11243 |
| `static_bfs_sharded_clock_tiny_lfu_s5000_d5000` | 4679.14 | 98.81 | 13.38 | 10456 |

## Split Sweep Findings

Best split by QPS within the non-admission hybrid policies:

| workload | split behavior |
| --- | --- |
| `original` | `s7500_d2500` wins 5 of 6 thread counts by QPS; `s2500_d7500` wins I/O/hit. Differences are small, and `StaticBFS` or 50/50 TinyLFU are often competitive. |
| `random_seed42` | `s7500_d2500` wins 4 of 6 thread counts by QPS; `s2500_d7500` wins I/O/hit. Differences are again small. |
| `kmeans_k100_seed42` | `s2500_d7500` wins all 6 thread counts by QPS, I/O, and hit rate. This workload needs more dynamic capacity. |

Best split by QPS within TinyLFU hybrid policies:

| workload | split behavior |
| --- | --- |
| `original` | 50/50 wins 4 of 6 thread counts; `s7500_d2500` wins 2 of 6. |
| `random_seed42` | `s2500_d7500` wins 3 of 6 and `s7500_d2500` wins 3 of 6. |
| `kmeans_k100_seed42` | `s2500_d7500` wins all 6 thread counts, but TinyLFU is still worse than no-admission CLOCK on this workload. |

## Workload-Level Aggregates

The table below averages each policy across the six thread counts. The speedup
column is the geometric mean of per-thread QPS speedup versus `NoCache`.

| workload | policy | static | dynamic | avg QPS | geomean speedup | avg I/O | avg hit % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | `StaticBFS` | 10000 | 0 | 2038.07 | 1.0961x | 104.32 | 8.56 |
| `original` | `sharded_clock` | 0 | 10000 | 1965.51 | 1.0411x | 104.44 | 8.45 |
| `original` | `sharded_clock_tiny_lfu` | 0 | 10000 | 2026.67 | 1.0786x | 102.68 | 9.99 |
| `original` | `static_bfs_sharded_clock_s2500_d7500` | 2500 | 7500 | 1994.27 | 1.0584x | 103.02 | 9.70 |
| `original` | `static_bfs_sharded_clock_s5000_d5000` | 5000 | 5000 | 1999.60 | 1.0669x | 103.19 | 9.54 |
| `original` | `static_bfs_sharded_clock_s7500_d2500` | 7500 | 2500 | 2007.83 | 1.0704x | 103.70 | 9.10 |
| `original` | `static_bfs_sharded_clock_tiny_lfu_s5000_d5000` | 5000 | 5000 | 2048.80 | 1.0956x | 102.87 | 9.83 |
| `random_seed42` | `StaticBFS` | 10000 | 0 | 2040.43 | 1.1004x | 104.32 | 8.56 |
| `random_seed42` | `sharded_clock` | 0 | 10000 | 1957.26 | 1.0402x | 105.33 | 7.67 |
| `random_seed42` | `sharded_clock_tiny_lfu` | 0 | 10000 | 2038.92 | 1.0923x | 102.86 | 9.83 |
| `random_seed42` | `static_bfs_sharded_clock_s2500_d7500` | 2500 | 7500 | 1979.34 | 1.0530x | 103.77 | 9.04 |
| `random_seed42` | `static_bfs_sharded_clock_s5000_d5000` | 5000 | 5000 | 1996.49 | 1.0680x | 103.81 | 9.00 |
| `random_seed42` | `static_bfs_sharded_clock_s7500_d2500` | 7500 | 2500 | 2014.43 | 1.0748x | 104.13 | 8.72 |
| `random_seed42` | `static_bfs_sharded_clock_tiny_lfu_s5000_d5000` | 5000 | 5000 | 2042.42 | 1.0933x | 103.02 | 9.69 |
| `kmeans_k100_seed42` | `StaticBFS` | 10000 | 0 | 2052.66 | 1.1074x | 104.32 | 8.56 |
| `kmeans_k100_seed42` | `sharded_clock` | 0 | 10000 | 2152.48 | 1.1540x | 85.14 | 25.37 |
| `kmeans_k100_seed42` | `sharded_clock_tiny_lfu` | 0 | 10000 | 2119.73 | 1.1338x | 89.44 | 21.59 |
| `kmeans_k100_seed42` | `static_bfs_sharded_clock_s2500_d7500` | 2500 | 7500 | 2193.73 | 1.1822x | 85.67 | 24.90 |
| `kmeans_k100_seed42` | `static_bfs_sharded_clock_s5000_d5000` | 5000 | 5000 | 2158.23 | 1.1602x | 87.95 | 22.90 |
| `kmeans_k100_seed42` | `static_bfs_sharded_clock_s7500_d2500` | 7500 | 2500 | 2101.67 | 1.1381x | 92.25 | 19.13 |
| `kmeans_k100_seed42` | `static_bfs_sharded_clock_tiny_lfu_s5000_d5000` | 5000 | 5000 | 2111.52 | 1.1400x | 92.10 | 19.27 |

## Interpretation

1. Equal-capacity hybrids are much more nuanced than the previous 20k hybrid
   results. The hybrid strategy is still useful, but the earlier large wins
   partly came from giving hybrid policies an extra 10,000-node dynamic tier.
2. On `original` and `random_seed42`, `StaticBFS` remains a very strong
   10,000-node baseline. Fair 50/50 CLOCK is close to it, while fair 50/50
   TinyLFU can slightly improve QPS and I/O at high concurrency.
3. On the locality-heavy `kmeans_k100_seed42` workload, dynamic capacity is
   more valuable than static capacity. `s2500_d7500` is the best hybrid split
   for every thread count and is slightly better than pure `sharded_clock` at
   32 threads.
4. TinyLFU admission is workload dependent. It helps `original` and
   `random_seed42` modestly, but hurts `kmeans_k100_seed42` by rejecting useful
   graph-locality nodes.
5. For fair 10,000-node comparisons, the default recommendation should be:
   keep `StaticBFS` and `sharded_clock` as primary baselines, use
   `static_bfs_sharded_clock_s2500_d7500` for locality-heavy workloads, and
   keep TinyLFU opt-in until the target workload shows both QPS and I/O gains.

## Recommendation Update

When reporting equal-capacity results, do not cite the 2026-07-03
`static_bfs_sharded_clock` numbers as a fair 10,000-node comparison. Cite this
run instead. The 2026-07-03 run should be described as:

```text
StaticBFS 10000 nodes + dynamic 10000 nodes = 20000 total hybrid nodes
```

This fair split run should be described as:

```text
StaticBFS X nodes + dynamic Y nodes = 10000 total hybrid nodes
```
