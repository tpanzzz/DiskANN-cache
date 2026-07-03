# Cache Policy Concurrency Strategy Plan

This note preserves the current cache-policy analysis and the implementation plan for
concurrent graph-ANNS cache work. It is intentionally separated from experiment
outputs so code changes, replay changes, and live benchmark reruns can be reviewed
independently.

## Experimental Evidence

The current evidence comes from the SIFT1M cache-evaluation artifacts under
`experiments/cache_eval/sift1m/results/`.

### One-thread live search, `search_list=100`, `beam_width=4`

Source:
`experiments/cache_eval/sift1m/results/query_workload_cache_policy_summary.csv`.

The main query-order result is that locality dominates dynamic-cache usefulness.

| workload | best current dynamic policy by QPS | QPS | hit rate | mean IO/query | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `original` | `TinyLFU(dynamic)` | 331.82 | 9.52% | 103.22 | Static BFS is still faster at 347.11 QPS. |
| `random_seed42` | `TinyLFU(dynamic)` | 334.54 | 9.36% | 103.40 | Static BFS is still faster at 347.14 QPS. |
| `kmeans_k100_seed42` | `CLOCK(dynamic)` | 383.62 | 36.98% | 71.89 | Query locality exposes a large dynamic-cache gain. |

For `kmeans_k100_seed42`, `CLOCK`, `FIFO`, and `LRU` converge near the same hit
rate range:

- `CLOCK(dynamic)`: 36.98% hit, 383.62 QPS, 71.89 mean IO/query.
- `FIFO(dynamic)`: 36.45% hit, 381.84 QPS, 72.50 mean IO/query.
- `LRU(dynamic)`: 37.09% hit, 358.25 QPS, 71.77 mean IO/query.

This is the strongest practical signal for `CLOCK`: it preserves nearly the same
hit rate as LRU on the locality workload while doing less metadata work on hits.

### Current policy sweep

Source:
`experiments/cache_eval/sift1m/results/current_cache_policy_summary.csv`.

With the original query order, dynamic policies reduce IO modestly but often lose
QPS to policy-maintenance overhead:

- `StaticBFS(original)`: 349.69 QPS, 8.56% hit, 104.32 mean IO/query.
- `FIFO(dynamic)`: 334.72 QPS, 8.58% hit, 104.29 mean IO/query.
- `LRU(dynamic)`: 329.34 QPS, 9.21% hit, 103.58 mean IO/query.
- `CLOCK(dynamic)`: 334.65 QPS, 9.03% hit, 103.78 mean IO/query.
- `TinyLFU(dynamic)`: 339.83 QPS, 9.52% hit, 103.22 mean IO/query.
- `W-TinyLFU(dynamic)`: 253.12 QPS, 10.36% hit, 102.26 mean IO/query.
- `LIRS(dynamic)`: 178.31 QPS, 10.46% hit, 102.14 mean IO/query.

The policies with the largest hit-rate improvements are not automatically the
fastest policies. For live graph search, every hit-path metadata update competes
with distance computation, disk-IO scheduling, and query-thread parallelism.

### BFS-warmed dynamic policies

Source:
`experiments/cache_eval/sift1m/results/warm_bfs_cache_policy_summary.csv`.

`DynamicNodeCacheWithBfsWarmup` is not the same as an immutable static BFS tier.
It initializes the dynamic payload and policy state from medoid-BFS nodes, then
allows normal replacement. In the current original-order run, warmup changes hit
rate only slightly:

- `CLOCK(dynamic+StaticBFS warmup)`: 9.04% hit versus 9.03% cold.
- `LRU(dynamic+StaticBFS warmup)`: 9.21% hit versus 9.21% cold.
- `TinyLFU(dynamic+StaticBFS warmup)`: 9.54% hit versus 9.52% cold.
- `Cacheus(dynamic+StaticBFS warmup)`: 10.69% hit versus 10.61% cold.

This supports keeping `StaticBFS(original)` as a separate immutable Tier 0
baseline instead of replacing it with dynamic warmup.

### Higher-recall live search

Source:
`experiments/cache_eval/sift1m/results/topk100_recall99_l1000_1thread_cold_cache_policy_summary.csv`.

At `search_list=1000`, `recall_at=100`, target recall 99%, all listed policies
reach 99.31% recall. Cache effects are smaller in percentage terms because the
query touches roughly 1011 nodes on average:

- `NoCache`: 37.78 QPS, 1011.06 mean IO/query.
- `StaticBFS(original)`: 38.14 QPS, 1.40% hit, 996.94 mean IO/query.
- `CLOCK(dynamic)`: 37.08 QPS, 3.82% hit, 972.46 mean IO/query.
- `TinyLFU(dynamic)`: 37.43 QPS, 3.86% hit, 972.03 mean IO/query.
- `W-TinyLFU(dynamic)`: 27.56 QPS, 4.61% hit, 964.42 mean IO/query.

The higher-recall run reinforces the same engineering conclusion: modest IO
reductions can be erased by expensive cache metadata.

### Offline Belady replay

Source:
`experiments/cache_eval/sift1m/results/workload_traces/belady_trace_replay_summary.csv`.

Belady replay provides an offline upper bound for the same cleaned access traces;
it does not represent a live policy. At capacity 10,000:

| workload | LRU hit rate | Belady hit rate | gap |
| --- | ---: | ---: | ---: |
| `original` | 9.21% | 24.70% | +15.49 pp |
| `random_seed42` | 7.91% | 23.57% | +15.66 pp |
| `kmeans_k100_seed42` | 37.09% | 47.12% | +10.03 pp |

Belady shows substantial remaining theoretical headroom, but the smaller gap on
the k-means-locality workload also shows that simple online policies already use
much of the exposed temporal locality.

## Policy Suitability

The practical concurrent-ANNS policy should optimize the product of hit rate,
hit-path cost, and lock behavior. The current data suggests the following
ranking for near-term engineering:

| policy family | suitability | reason |
| --- | --- | --- |
| `StaticBFS(original)` | required baseline and Tier 0 | Immutable, no query-time lock, captures graph-entry hot roots. |
| `CLOCK` | best current dynamic starting point | k-means locality gives 36.98% hit and the highest dynamic QPS. |
| `FIFO` | useful low-overhead baseline | Hit path does not mutate replacement metadata. |
| `LRU` | useful only after O(1) metadata | Hit rate is strong, but current `VecDeque` removal is linear. |
| TinyLFU/SIEVE-like admission | useful as an optional filter | Best treated as admission around CLOCK/FIFO/LRU, not as a costly full replacement path. |
| `LFU`, `GDSF`, `LeCaR` | secondary | Need heap compaction or indexed metadata; more hit-path work. |
| `2Q`, `SLRU`, `W-TinyLFU`, `Cacheus` | secondary | Segment movement and history maintenance are expensive under one global lock. |
| `LIRS`, `ARC` | experimental | Interesting semantics but not current QPS leaders; require more structural cleanup before concurrent recommendation. |

## Why Dynamic Policies Lose QPS

Dynamic policies reduce mean IO/query, but the live runs show that a lower IO
count does not guarantee higher QPS. The current implementation pays several
costs for every dynamic cache lookup:

1. A global `Arc<Mutex<DynamicNodeCache<Data>>>` serializes all query threads.
2. Dynamic hits clone the entire `CachedNode`, including the vector and adjacency
   list, into `cached_nodes_for_current_read`.
3. Several policies use `VecDeque` plus linear removal on hit, promotion,
   demotion, ghost-history updates, or eviction cleanup.
4. Heap-backed policies (`LFU`, `GDSF`, `LeCaR`, `Cacheus`) use lazy stale-entry
   skipping and can accumulate large heaps.
5. Sketch-backed policies update frequency state on most accesses.
6. Trace replay measures replacement semantics without disk-provider overhead,
   thread contention, payload clone cost, or live search scheduling effects.

These costs explain why `W-TinyLFU`, `LIRS`, `2Q`, and `Cacheus` can have higher
hit rates but lower QPS than simpler policies.

## Concurrency Bottlenecks

The current live path is:

- static cache: immutable `Arc<Cache<Data>>`, read without locking;
- dynamic cache: one `Arc<Mutex<DynamicNodeCache<Data>>>`;
- dynamic hit: lock cache, call `lookup`, deep clone a `CachedNode`, unlock, then
  keep the clone in the provider's current-read map;
- dynamic miss: read from disk, construct a `CachedNode`, lock the same global
  cache again, update policy state, possibly evict, and store the payload.

The main bottlenecks are therefore:

- global lock contention across all query threads;
- lock hold time inflated by payload clone work;
- policy state structures that mutate on every hit;
- linear queue deletion in replacement metadata;
- replay/live mismatch when interpreting hit rate alone.

## Recommended Strategy

The recommended final strategy for concurrent graph ANNS is:

**StaticBFS + Sharded CLOCK + optional TinyLFU/SIEVE-like admission.**

Rationale:

- Static BFS captures medoid/root-neighborhood hotness with an immutable Tier 0.
- CLOCK has a short hit path and reached the best dynamic QPS on the
  locality-rich workload.
- Sharding replaces one global lock with many short shard-local locks.
- Shard-local eviction is acceptable for graph search because IDs are numerous
  and accesses are naturally distributed; any hit-rate loss must be measured.
- TinyLFU/SIEVE-like admission can filter one-off tail nodes when it improves
  QPS or IO; it should remain optional and disabled by default until live
  concurrent benchmarks justify enabling it.

The current-policy default recommendation is `CLOCK`. The improved default
recommendation is `StaticBFS + Sharded CLOCK`, with `CLOCK + admission` evaluated
as an opt-in variant.

## Per-Policy Locking Plan

### StaticBFS

- Keep as immutable `Arc<Cache<Data>>`.
- Do not add query-time locks.
- In hybrid mode, check StaticBFS first and dynamic cache second.

### FIFO

- Use shard-local locks.
- Do not update replacement metadata on hits.
- Update queue only on admission and eviction.
- Keep as a low-overhead concurrent baseline.

### CLOCK

- Use shard-local locks around ring/hand/ref-bit state.
- On hit, set only the reference bit.
- On eviction, advance only the shard-local hand.
- Treat as the main concurrent policy.

### LRU

- Replace `VecDeque` linear deletion before treating LRU as a primary concurrent
  policy.
- Use an O(1) linked metadata structure: resident id to list node, plus
  predecessor/successor storage or an equivalent linked hash map.
- Keep shard-local LRU state.

### TinyLFU / SIEVE-like Admission

- Do not make admission the only replacement policy in the concurrent path.
- Use it as an optional filter around CLOCK/FIFO/LRU.
- Keep sketch or seen-state shard-local, or use thread-local buffers with batch
  merge in a later iteration.
- Avoid a global sketch lock on every hit.

### LFU / GDSF / LeCaR

- Keep frequency state shard-local.
- Add heap compaction when lazy heaps exceed a multiple of resident entries,
  for example `heap.len() > 4 * resident.len()`.
- Keep LeCaR experts shard-local first; evaluate global weights only after
  contention measurements justify the additional synchronization.

### 2Q / SLRU / W-TinyLFU / Cacheus

- Keep all segments shard-local.
- Replace segment queues with O(1) deletion/movement before concurrent
  recommendation.
- Preserve current approximations unless a separate paper-replication task
  requires exact adaptive variants.

### ARC / LIRS

- Preserve as experimental policies.
- Optimize queue operations before using them in concurrent live-search sweeps.
- Do not recommend them unless new experiments beat CLOCK-like alternatives.

## Implementation Roadmap

Use separate git branches and commits inside the `DiskANN/` repository:

1. `docs/cache-policy-concurrency-plan`
   - Add this document only.
   - Commit: `Document cache policy concurrency strategy`.
2. `opt/belady-replay-heap`
   - Keep the heap-based `replay_belady_optimal` implementation.
   - Keep `belady_counts_expected_hits`.
   - Commit: `Optimize Belady trace replay`.
3. `opt/cache-payload-arc`
   - Store dynamic payloads as `Arc<CachedNode<Data>>`.
   - Return cloned `Arc` handles on dynamic hits.
   - Keep current query references valid after eviction.
   - Commit: `Use shared cached node handles for dynamic cache hits`.
4. `opt/sharded-cache-core`
   - Add shard capacity splitting, stable shard mapping, stats aggregation, and
     `--cache_backend global|sharded`, `--cache_shards <usize>` CLI plumbing.
   - Commit: `Add sharded dynamic cache core`.
5. `opt/sharded-clock-fifo`
   - Validate shard-local FIFO and CLOCK behavior.
   - Commit: `Add sharded CLOCK and FIFO cache policies`.
6. `opt/o1-lru`
   - Replace LRU linear metadata with O(1) movement/deletion.
   - Commit: `Use O(1) metadata for LRU cache policy`.
7. `opt/admission-policies`
   - Add optional TinyLFU/SIEVE-like admission around dynamic policies.
   - Commit: `Add lightweight admission for dynamic cache policies`.
8. `opt/frequency-policies`
   - Add heap compaction or indexed metadata for LFU/GDSF/LeCaR.
   - Commit: `Optimize frequency-based cache policies`.
9. `opt/segmented-policies`
   - Optimize segment queues and shard-local locking for 2Q/SLRU/W-TinyLFU/Cacheus.
   - Commit: `Optimize segmented cache policies`.
10. `eval/concurrent-cache-policies`
   - Add concurrent evaluation scripts, CSV summaries, and experiment notes.
   - Commit: `Evaluate concurrent cache policies`.

Do not mix unrelated policy families in the same commit. Do not overwrite
datasets or generated index artifacts.

## Test And Evaluation Plan

### Unit tests

- Capacity split sums to total capacity and assigns the remainder to early shards.
- The same vertex id always maps to the same shard.
- Shard-local FIFO/CLOCK never exceed capacity.
- CLOCK hit sets the reference bit and victim selection skips second-chance nodes.
- LRU O(1) metadata preserves old replay semantics.
- `Arc<CachedNode>` handles remain readable after eviction removes the resident
  map entry.
- TinyLFU/SIEVE admission can reject one-time low-frequency candidates.
- Aggregated sharded stats equal the sum of per-shard stats.
- Heap-based Belady replay preserves known hit/miss counts.

### Replay tests

Use cleaned traces for:

- `original`
- `random_seed42`
- `kmeans_k100_seed42`

Compare:

- `original` static/no-cache live baselines where applicable;
- old global `fifo`, `lru`, `clock`;
- new sharded `fifo`, `lru`, `clock`;
- new `clock` plus admission;
- Belady upper bound.

Use capacities `1000`, `5000`, `10000`, and `20000`. Document any expected hit
rate difference caused by shard-local capacity partitioning.

### Live search tests

Fixed parameters:

- dataset: SIFT1M;
- `search_list=100`;
- `beam_width=4`;
- `recall_at=10`;
- `search_io_limit=1000`;
- `cache_capacity=10000`;
- static cache nodes: `10000` when evaluating the hybrid Tier 0;
- workloads: `original`, `random_seed42`, `kmeans_k100_seed42`;
- threads: `1`, `2`, `4`, `8`, `16`, `32`.

Policies:

- `NoCache`;
- `StaticBFS(original)`;
- current global `CLOCK`;
- current global `LRU`;
- sharded `CLOCK`;
- sharded `FIFO`;
- sharded `LRU`;
- sharded `CLOCK + TinyLFU`;
- `StaticBFS + sharded CLOCK`;
- `StaticBFS + sharded CLOCK + TinyLFU`.

CSV fields should include:

- workload;
- policy;
- backend;
- shards;
- admission;
- threads;
- cache capacity nodes;
- static cache nodes;
- QPS;
- mean latency and p999 latency;
- mean IO/query;
- total, static, and dynamic hit rates;
- admission rejection rate;
- lock wait/hold time if instrumented;
- recall.

### Acceptance criteria

- Recall matches current experiments within normal floating-point noise.
- Single-thread sharded CLOCK is not materially slower than current global CLOCK.
- At 8+ threads, sharded CLOCK materially improves QPS over global-mutex
  CLOCK/LRU.
- On `kmeans_k100_seed42`, sharded CLOCK stays near the current 36-37% dynamic
  hit-rate range, unless documented shard-local capacity effects explain the
  difference.
- `StaticBFS + sharded CLOCK` is not materially slower than immutable StaticBFS
  on original/random workloads.
- Admission remains disabled by default unless it improves live QPS or IO.
