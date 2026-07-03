# Concurrent Cache Policy Results, 2026-07-03

This document summarizes the full SIFT1M live-search sweep for the concurrent
cache-policy optimizations on branch `eval/concurrent-cache-policies`.

## Run Configuration

The sweep was run from the `DiskANN/` repository root:

```bash
RESULT_ROOT=experiments/cache_eval/sift1m/results/concurrent_cache_full_20260703 \
THREADS_LIST="1 2 4 8 16 32" \
WORKLOADS_LIST="original random_seed42 kmeans_k100_seed42" \
experiments/cache_eval/scripts/run_concurrent_cache_policy_sweep.sh
```

The summary CSV was generated with:

```bash
python3 experiments/cache_eval/scripts/summarize_concurrent_cache_results.py \
  --manifest experiments/cache_eval/sift1m/results/concurrent_cache_full_20260703/run_manifest.csv \
  --output experiments/cache_eval/sift1m/results/concurrent_cache_full_20260703/concurrent_cache_policy_summary.csv \
  --dedupe-latest
```

Dataset and search settings:

- dataset: SIFT1M;
- index prefix: `experiments/cache_eval/sift1m/sift_disk`;
- workloads: `original`, `random_seed42`, `kmeans_k100_seed42`;
- threads: `1, 2, 4, 8, 16, 32`;
- policies per workload/thread: `NoCache`, `StaticBFS`, `global_clock`,
  `global_lru`, `sharded_clock`, `sharded_fifo`, `sharded_lru`,
  `sharded_clock_tiny_lfu`, `static_bfs_sharded_clock`,
  `static_bfs_sharded_clock_tiny_lfu`;
- `search_list=100`, `beam_width=4`, `recall_at=10`,
  `search_io_limit=1000`;
- dynamic cache capacity: 10,000 nodes;
- StaticBFS tier size: 10,000 nodes;
- sharded cache: 64 shards.

Machine context during this run:

- logical CPUs: 112;
- memory: 503 GiB total, 394 GiB available after the run.

## Output Files

Primary outputs:

- manifest:
  `experiments/cache_eval/sift1m/results/concurrent_cache_full_20260703/run_manifest.csv`;
- consolidated summary:
  `experiments/cache_eval/sift1m/results/concurrent_cache_full_20260703/concurrent_cache_policy_summary.csv`;
- per-run logs:
  `experiments/cache_eval/sift1m/results/concurrent_cache_full_20260703/logs/`;
- derived best-policy summary:
  `experiments/cache_eval/sift1m/results/concurrent_cache_full_20260703/best_policy_by_workload_thread.csv`;
- derived pairwise comparisons:
  `experiments/cache_eval/sift1m/results/concurrent_cache_full_20260703/key_policy_comparisons.csv`;
- derived workload-policy aggregate:
  `experiments/cache_eval/sift1m/results/concurrent_cache_full_20260703/policy_aggregate_by_workload.csv`.

`TRACE_CACHE_ACCESS=1` was intentionally not enabled for this full sweep, to
avoid very large JSONL trace output. Therefore the summary CSV does not split
hits into static-tier and dynamic-tier rates. Live output also does not yet emit
admission-reject counts or lock wait/hold timing, so those fields remain blank.

## Completeness Checks

The sweep completed successfully.

| check | result |
| --- | ---: |
| expected rows | 180 |
| summarized rows | 180 |
| status `ok` rows | 180 |
| failed rows | 0 |
| workload/thread groups | 18 |
| policies per workload/thread group | 10 |
| Recall@10 range | 97.04% to 97.04% |

Recall stayed identical across all cache variants in this run. The observed
performance differences are therefore cache-path and I/O differences, not
recall/quality changes.

## Best Policy By Workload And Thread

| workload | threads | best policy | QPS | mean IO/query | cache hit % | QPS vs NoCache |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| original | 1 | StaticBFS | 340.74 | 104.32 | 8.56 | 1.0953x |
| original | 2 | StaticBFS | 656.93 | 104.32 | 8.56 | 1.1234x |
| original | 4 | static_bfs_sharded_clock_tiny_lfu | 1208.53 | 100.90 | 11.55 | 1.0625x |
| original | 8 | static_bfs_sharded_clock_tiny_lfu | 2176.96 | 100.84 | 11.60 | 1.1213x |
| original | 16 | static_bfs_sharded_clock_tiny_lfu | 3412.64 | 100.94 | 11.51 | 1.1181x |
| original | 32 | static_bfs_sharded_clock_tiny_lfu | 4598.05 | 100.94 | 11.52 | 1.1645x |
| random_seed42 | 1 | StaticBFS | 341.93 | 104.32 | 8.56 | 1.0880x |
| random_seed42 | 2 | static_bfs_sharded_clock_tiny_lfu | 657.40 | 101.14 | 11.34 | 1.1058x |
| random_seed42 | 4 | static_bfs_sharded_clock_tiny_lfu | 1222.27 | 101.10 | 11.38 | 1.0947x |
| random_seed42 | 8 | StaticBFS | 2144.46 | 104.32 | 8.56 | 1.1056x |
| random_seed42 | 16 | static_bfs_sharded_clock_tiny_lfu | 3393.53 | 101.10 | 11.38 | 1.1101x |
| random_seed42 | 32 | static_bfs_sharded_clock_tiny_lfu | 4559.92 | 101.12 | 11.36 | 1.1640x |
| kmeans_k100_seed42 | 1 | static_bfs_sharded_clock | 374.80 | 70.47 | 38.22 | 1.1976x |
| kmeans_k100_seed42 | 2 | static_bfs_sharded_clock | 699.06 | 76.52 | 32.92 | 1.1656x |
| kmeans_k100_seed42 | 4 | static_bfs_sharded_clock_tiny_lfu | 1267.65 | 86.41 | 24.25 | 1.1476x |
| kmeans_k100_seed42 | 8 | static_bfs_sharded_clock | 2296.51 | 86.85 | 23.87 | 1.1732x |
| kmeans_k100_seed42 | 16 | static_bfs_sharded_clock | 3604.93 | 89.43 | 21.61 | 1.1804x |
| kmeans_k100_seed42 | 32 | static_bfs_sharded_clock | 4967.61 | 90.96 | 20.27 | 1.2663x |

## High-Concurrency Snapshot, 32 Threads

### `original`

| policy | QPS | mean IO/query | cache hit % | p99.9 latency us |
| --- | ---: | ---: | ---: | ---: |
| NoCache | 3948.40 | 114.08 | 0.00 | 11223 |
| StaticBFS | 4478.96 | 104.32 | 8.56 | 13248 |
| global_clock | 3092.00 | 105.07 | 7.89 | 14741 |
| global_lru | 2721.23 | 104.80 | 8.13 | 18225 |
| sharded_clock | 4373.50 | 105.09 | 7.88 | 13475 |
| sharded_fifo | 4282.81 | 105.67 | 7.37 | 15933 |
| sharded_lru | 4367.37 | 104.84 | 8.10 | 14364 |
| sharded_clock_tiny_lfu | 4495.55 | 102.70 | 9.97 | 12744 |
| static_bfs_sharded_clock | 4545.87 | 101.89 | 10.68 | 12170 |
| static_bfs_sharded_clock_tiny_lfu | 4598.05 | 100.94 | 11.52 | 11477 |

### `random_seed42`

| policy | QPS | mean IO/query | cache hit % | p99.9 latency us |
| --- | ---: | ---: | ---: | ---: |
| NoCache | 3917.51 | 114.08 | 0.00 | 14190 |
| StaticBFS | 4470.59 | 104.32 | 8.56 | 11050 |
| global_clock | 2860.79 | 105.23 | 7.76 | 15971 |
| global_lru | 2211.57 | 105.08 | 7.89 | 21259 |
| sharded_clock | 4389.44 | 105.31 | 7.69 | 11536 |
| sharded_fifo | 4352.86 | 105.76 | 7.29 | 9463 |
| sharded_lru | 4354.46 | 105.06 | 7.90 | 12512 |
| sharded_clock_tiny_lfu | 4478.43 | 102.86 | 9.83 | 12020 |
| static_bfs_sharded_clock | 4506.98 | 102.18 | 10.43 | 11304 |
| static_bfs_sharded_clock_tiny_lfu | 4559.92 | 101.12 | 11.36 | 12578 |

### `kmeans_k100_seed42`

| policy | QPS | mean IO/query | cache hit % | p99.9 latency us |
| --- | ---: | ---: | ---: | ---: |
| NoCache | 3922.81 | 114.08 | 0.00 | 13307 |
| StaticBFS | 4476.93 | 104.32 | 8.56 | 12052 |
| global_clock | 2839.76 | 93.50 | 18.04 | 16482 |
| global_lru | 1968.96 | 93.31 | 18.21 | 23304 |
| sharded_clock | 4820.69 | 93.59 | 17.96 | 11713 |
| sharded_fifo | 4780.68 | 94.41 | 17.24 | 13207 |
| sharded_lru | 4830.42 | 93.23 | 18.28 | 11362 |
| sharded_clock_tiny_lfu | 4702.52 | 97.00 | 14.97 | 12530 |
| static_bfs_sharded_clock | 4967.61 | 90.96 | 20.27 | 8620 |
| static_bfs_sharded_clock_tiny_lfu | 4769.03 | 95.41 | 16.36 | 11724 |

## Sharded CLOCK Versus Global CLOCK

This comparison isolates the sharded backend while keeping the replacement
policy as CLOCK.

| workload | threads | QPS delta | QPS ratio | mean IO delta | hit delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 1 | +0.39% | 1.0039x | +0.00 | +0.00 pp |
| original | 2 | +4.50% | 1.0450x | +0.03 | -0.03 pp |
| original | 4 | -3.06% | 0.9694x | +0.02 | -0.01 pp |
| original | 8 | +0.08% | 1.0008x | -0.08 | +0.07 pp |
| original | 16 | +13.62% | 1.1362x | -0.03 | +0.02 pp |
| original | 32 | +41.45% | 1.4145x | +0.02 | -0.01 pp |
| random_seed42 | 1 | -1.32% | 0.9868x | +0.01 | -0.01 pp |
| random_seed42 | 2 | -5.10% | 0.9490x | +0.01 | +0.00 pp |
| random_seed42 | 4 | +3.16% | 1.0316x | -0.01 | +0.02 pp |
| random_seed42 | 8 | +4.88% | 1.0488x | +0.00 | +0.00 pp |
| random_seed42 | 16 | +13.04% | 1.1304x | -0.01 | +0.01 pp |
| random_seed42 | 32 | +53.43% | 1.5343x | +0.08 | -0.07 pp |
| kmeans_k100_seed42 | 1 | -0.11% | 0.9989x | +0.03 | -0.02 pp |
| kmeans_k100_seed42 | 2 | +1.83% | 1.0183x | +0.03 | -0.03 pp |
| kmeans_k100_seed42 | 4 | +0.48% | 1.0048x | +0.03 | -0.03 pp |
| kmeans_k100_seed42 | 8 | +4.02% | 1.0402x | +0.31 | -0.27 pp |
| kmeans_k100_seed42 | 16 | +29.43% | 1.2943x | -0.21 | +0.18 pp |
| kmeans_k100_seed42 | 32 | +69.76% | 1.6976x | +0.09 | -0.08 pp |

Sharding is neutral to mildly noisy at one to four threads, then becomes clearly
beneficial at higher concurrency. At 32 threads, sharded CLOCK improves QPS by
41.45% on `original`, 53.43% on `random_seed42`, and 69.76% on
`kmeans_k100_seed42`, with essentially unchanged mean IO and hit rate. This is
the clearest evidence that the global mutex was a high-concurrency bottleneck.

## Sharded LRU Versus Global LRU

| workload | 1t | 2t | 4t | 8t | 16t | 32t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | +1.2% | +2.8% | -2.3% | +14.3% | +22.4% | +60.5% |
| random_seed42 | -0.9% | +1.6% | +7.9% | +13.5% | +21.7% | +96.9% |
| kmeans_k100_seed42 | -1.1% | -3.8% | +9.1% | +19.4% | +43.4% | +145.3% |

The O(1) metadata plus sharded backend makes LRU viable at high thread counts.
However, CLOCK remains the simpler default because it reaches similar hit rates
with less hit-path mutation and fewer policy-specific moving parts.

## TinyLFU Admission Versus Sharded CLOCK

This comparison uses the same sharded CLOCK backend and adds TinyLFU admission.

| workload | threads | QPS delta | QPS ratio | mean IO delta | hit delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 1 | +3.64% | 1.0364x | -1.19 | +1.04 pp |
| original | 2 | +4.12% | 1.0412x | -1.41 | +1.24 pp |
| original | 4 | +1.59% | 1.0159x | -1.73 | +1.52 pp |
| original | 8 | +8.21% | 1.0821x | -1.93 | +1.70 pp |
| original | 16 | +2.90% | 1.0290x | -2.11 | +1.85 pp |
| original | 32 | +2.79% | 1.0279x | -2.39 | +2.09 pp |
| random_seed42 | 1 | +6.20% | 1.0620x | -2.44 | +2.14 pp |
| random_seed42 | 2 | +7.55% | 1.0755x | -2.42 | +2.12 pp |
| random_seed42 | 4 | +5.72% | 1.0572x | -2.44 | +2.13 pp |
| random_seed42 | 8 | +5.03% | 1.0503x | -2.42 | +2.12 pp |
| random_seed42 | 16 | +4.99% | 1.0499x | -2.46 | +2.16 pp |
| random_seed42 | 32 | +2.03% | 1.0203x | -2.45 | +2.14 pp |
| kmeans_k100_seed42 | 1 | -4.45% | 0.9555x | +7.80 | -6.84 pp |
| kmeans_k100_seed42 | 2 | -2.31% | 0.9769x | +5.72 | -5.01 pp |
| kmeans_k100_seed42 | 4 | +2.52% | 1.0252x | +2.99 | -2.62 pp |
| kmeans_k100_seed42 | 8 | +1.07% | 1.0107x | +3.23 | -2.83 pp |
| kmeans_k100_seed42 | 16 | -1.64% | 0.9836x | +4.20 | -3.69 pp |
| kmeans_k100_seed42 | 32 | -2.45% | 0.9755x | +3.41 | -2.99 pp |

TinyLFU admission helps the original and random workloads by filtering some
tail churn and reducing mean IO by roughly 1.2 to 2.5 IO/query. It hurts the
locality-rich k-means workload by rejecting nodes that are reused soon, reducing
hit rate by roughly 2.6 to 6.8 percentage points. Admission should remain
optional and workload-dependent, not the default concurrent cache setting.

## StaticBFS Plus Sharded CLOCK Versus StaticBFS

This comparison isolates the hybrid Tier 0 + Tier 1 cache without TinyLFU
admission.

| workload | threads | QPS delta | QPS ratio | mean IO delta | hit delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 1 | -3.71% | 0.9629x | -3.58 | +3.13 pp |
| original | 2 | -3.88% | 0.9612x | -3.30 | +2.88 pp |
| original | 4 | -3.28% | 0.9672x | -3.07 | +2.68 pp |
| original | 8 | -1.37% | 0.9863x | -2.77 | +2.42 pp |
| original | 16 | -0.25% | 0.9975x | -2.67 | +2.33 pp |
| original | 32 | +1.49% | 1.0149x | -2.43 | +2.12 pp |
| random_seed42 | 1 | -4.09% | 0.9591x | -2.15 | +1.87 pp |
| random_seed42 | 2 | -7.38% | 0.9262x | -2.12 | +1.85 pp |
| random_seed42 | 4 | -5.21% | 0.9479x | -2.14 | +1.87 pp |
| random_seed42 | 8 | -3.96% | 0.9604x | -2.09 | +1.83 pp |
| random_seed42 | 16 | -1.80% | 0.9820x | -2.15 | +1.87 pp |
| random_seed42 | 32 | +0.81% | 1.0081x | -2.14 | +1.87 pp |
| kmeans_k100_seed42 | 1 | +9.17% | 1.0917x | -33.85 | +29.66 pp |
| kmeans_k100_seed42 | 2 | +8.60% | 1.0860x | -27.80 | +24.36 pp |
| kmeans_k100_seed42 | 4 | +2.83% | 1.0283x | -21.43 | +18.78 pp |
| kmeans_k100_seed42 | 8 | +8.09% | 1.0809x | -17.47 | +15.31 pp |
| kmeans_k100_seed42 | 16 | +7.66% | 1.0766x | -14.89 | +13.05 pp |
| kmeans_k100_seed42 | 32 | +10.96% | 1.1096x | -13.36 | +11.71 pp |

The hybrid strongly helps the locality-rich workload and becomes neutral to
mildly positive on original/random at 32 threads. At lower thread counts on
original/random, the extra dynamic-tier lookup cost can exceed the modest I/O
reduction. This supports keeping immutable `StaticBFS` as the baseline and using
`StaticBFS + sharded CLOCK` when query locality or high concurrency justifies
the dynamic tier.

## Interpretation

The full live-search data supports the concurrency strategy:

1. `StaticBFS` remains a strong baseline for non-local or weak-locality query
   orders because it is immutable and lock-free during search.
2. Dynamic cache value depends heavily on query locality. On `kmeans_k100_seed42`,
   dynamic CLOCK/LRU cut mean IO sharply, from 114.08 to about 71.8-72.0 in the
   one-thread case.
3. Sharding is the main high-concurrency optimization. It preserves hit rate and
   mean IO while removing the severe global-lock QPS collapse at 16 and 32
   threads.
4. O(1) LRU metadata plus sharding fixes the worst high-thread LRU behavior, but
   CLOCK is still the preferred default because it is simpler and competitive.
5. TinyLFU admission is mixed. It improves original/random workloads but hurts
   the k-means locality workload. It should stay opt-in.
6. The best overall practical strategy after this run is still
   `StaticBFS + sharded CLOCK`, with TinyLFU admission enabled only after a
   workload-specific validation run.

## Acceptance Results

| acceptance item | result |
| --- | --- |
| Recall@10 remains consistent | Passed: all 180 rows are 97.04%. |
| Sharded CLOCK close to global CLOCK at one thread | Passed: -1.32% to +0.39% QPS delta across workloads. |
| Sharded CLOCK improves high-thread QPS | Passed: at 32 threads, +41.45% to +69.76% versus global CLOCK. |
| K-means locality gives strong dynamic-cache hit rate | Passed in one-thread runs: CLOCK/LRU near 37%; high-thread live runs have lower hit rate because concurrent query interleaving changes cache state. |
| StaticBFS + sharded CLOCK not materially worse at high concurrency | Passed at 32 threads: +1.49% original, +0.81% random, +10.96% k-means versus StaticBFS. |
| Admission default should be justified by data | Passed as a negative result: TinyLFU hurts k-means hit rate and should remain opt-in. |

## Limitations

- Each matrix cell was run once; the table does not include variance or
  confidence intervals.
- Per-tier static/dynamic hit split was not collected because cache-source
  tracing was disabled for the full run.
- Admission reject percentage and lock wait/hold timing are not emitted by the
  live-search binary yet.
- Results are for SIFT1M on one high-core-count machine. Larger datasets and
  lower-recall/higher-recall operating points should be rerun before claiming a
  general result.

## Recommendation

Use the following order for subsequent experiments and paper-quality
evaluation:

1. Keep `StaticBFS` as the no-lock hot-root baseline.
2. Use `sharded_clock` as the primary dynamic-cache baseline.
3. Use `static_bfs_sharded_clock` as the recommended hybrid strategy for
   locality-aware or high-concurrency workloads.
4. Keep `static_bfs_sharded_clock_tiny_lfu` and `sharded_clock_tiny_lfu` as
   opt-in variants; do not make admission the default until a target workload
   shows both QPS and I/O gains.
5. Report the negative admission result on `kmeans_k100_seed42`, because it is
   important evidence that higher admission selectivity can reject useful graph
   locality.
