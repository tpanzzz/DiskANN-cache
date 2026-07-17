# Spatial-locality pilot results, 2026-07-17

## 1. Scope

This pilot tests three separate questions that must not be conflated:

1. Do base and query vectors occupy similar regions of vector space?
2. Does the released query row order contain meaningful short-range locality?
3. At Recall@10 above 95%, how concentrated are DiskANN graph-node expansions?

The analysis uses `K=100`, seed 42, one search thread, and the complete released
query split for every dataset. Generated tables and plots are under:

```text
experiments/cache_eval/results/vector_patterns/
experiments/cache_eval/results/spatial_locality_pilot/
```

The ANN-Benchmarks HDF5 files call the base vectors `train`. They do not expose
a separate learn-query split, so this pilot compares base and query. The tools
support an additional `learn_key` when a future catalog entry provides one.

## 2. Datasets

| Dataset | Base vectors | Queries | Dimensions | Distance | Released query-order interpretation |
|---|---:|---:|---:|---|---|
| SIFT1M | 1,000,000 | 10,000 | 128 | L2 | TexMex file order; no documented request timestamps |
| Fashion-MNIST | 60,000 | 10,000 | 784 | L2 | Official test-file order, not request order |
| GloVe-25 | 1,183,514 | 10,000 | 25 | cosine | Deterministic randomized train/test split |
| Last.fm ANN | 292,385 | 50,000 | 65 | cosine | ALS user-factor row order, not listening-event order |

Last.fm has 65 dimensions because ANN-Benchmarks transforms the original
maximum-inner-product factors into a cosine-search representation.

## 3. Query sequence locality

Queries were assigned to the independently fitted query clusters without
changing row order. Lag-1 same-cluster probability was compared with ten
fixed-seed permutations of the same labels.

| Dataset | Observed lag-1 | Shuffled lag-1 | Lift | Interpretation |
|---|---:|---:|---:|---|
| SIFT1M | 0.020002 | 0.011441 | 1.7483x | File order has detectable grouping, but no known temporal semantics |
| Fashion-MNIST | 0.012101 | 0.011551 | 1.0476x | Effectively shuffled at lag 1 |
| GloVe-25 | 0.011301 | 0.011181 | 1.0107x | Expected from randomized split |
| Last.fm ANN | 0.009740 | 0.011006 | 0.8850x | No positive sequential locality |

The main conclusion is negative but important: these standard benchmark query
arrays are useful as query sets, but they are not production arrival traces.
K-means sorting is a controlled locality intervention, not a reconstruction of
their original request order.

## 4. Base-query spatial-pattern similarity

Two complementary analyses were run.

### 4.1 Independent center matching

Base and query were clustered independently. Cluster centers were matched with
the Hungarian algorithm. Center distance is normalized by the pooled cluster
radius.

| Dataset | Center distance/radius | Matched mass Spearman | Matched mass JS divergence |
|---|---:|---:|---:|
| SIFT1M | 0.4760 | 0.4995 | 0.0189 |
| Fashion-MNIST | 0.4558 | 0.6977 | 0.0180 |
| GloVe-25 | 0.4930 | 0.4862 | 0.0179 |
| Last.fm ANN | 3.0602 | 0.0636 | 0.1660 |

### 4.2 Query occupancy on base-fitted centers

The base centers were kept fixed and every query was assigned to its nearest
base center. This is the more direct test of whether base-vector clusters can
bootstrap a query router.

| Dataset | Occupancy Pearson | Occupancy Spearman | Occupancy JS divergence |
|---|---:|---:|---:|
| SIFT1M | 0.9335 | 0.9334 | 0.002589 |
| Fashion-MNIST | 0.9558 | 0.9543 | 0.002086 |
| GloVe-25 | 0.9589 | 0.9418 | 0.001565 |
| Last.fm ANN | -0.0760 | -0.7481 | 0.789898 |

For SIFT, Fashion-MNIST, and GloVe, base geometry is a strong cold-start prior
for aggregate query-region occupancy even though independently numbered cluster
masses correlate only moderately. Last.fm is the counterexample: item factors
form the base and user factors form the queries, so clustering graph nodes is a
poor substitute for clustering demand.

This result argues for the following prototype source order:

1. historical production queries;
2. representative offline learn queries;
3. training queries from the retrieval model;
4. base-vector samples only as a validated cold-start fallback.

## 5. High-recall DiskANN configurations

All trace searches use `num_threads=1`, `beam_width=4`, `recall_at=10`, an LRU
cache of capacity one, and a cache-access trace. Capacity one preserves the
search path while causing the provider to emit every requested graph node.

| Dataset | Index/search configuration | Selected L | Mean loads/query | Recall@10 |
|---|---|---:|---:|---:|
| SIFT1M | existing R=64/Lbuild=100, PQ=16, I/O limit=1000 | 100 | 114.07 | 97.04% |
| Fashion-MNIST | R=64/Lbuild=100, PQ=49, I/O limit=1000 | 50 | 63.00 | 99.13% |
| GloVe-25 | R=96/Lbuild=200, PQ=5, I/O limit=2000 | 1600 | 1609.88 | 95.94% |
| Last.fm ANN | R=96/Lbuild=200, PQ=16, I/O limit=1000 | 100 | 122.69 | 95.79% |

GloVe required a much larger search budget: L=1200 reached only 94.69%, while
L=1600 reached 95.94%. This makes GloVe's expansion count qualitatively
different from the other datasets and should be considered when comparing raw
expansion totals.

A follow-up identified that the raw GloVe vectors were not unit normalized,
while the current PQ navigation path evaluates cosine candidates with squared
L2. The L=1600 trace remains correct for that exact configuration, but it is not
the recommended GloVe setup. After normalization, PQ16 reaches 96.79% at L=20;
exact in-memory cosine navigation reaches 95.25% at L=10 on the raw graph. See
`docs/glove_pq_full_precision_results_20260717.md`.

Last.fm exposed a separate parameter sensitivity. With only five PQ chunks,
Recall@10 was 44.83%, 61.27%, and 77.50% at L=50, 100, and 200. Keeping the
same graph parameters but increasing PQ chunks to 16 raised recall to 87.66%,
95.79%, and 98.90%. The selected expansion trace therefore uses PQ=16 and
L=100.

## 6. Node-expansion concentration

| Dataset | Total expansions | Expanded graph nodes | Gini | Top 1% share | Top 5% share | Nodes for 50% mass | Nodes for 90% mass | Nodes for 95% mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIFT1M | 1,140,710 | 52.51% | 0.6788 | 14.54% | 31.49% | 11.50% | 41.10% | 46.80% |
| Fashion-MNIST | 630,043 | 91.68% | 0.5492 | 18.07% | 29.92% | 15.62% | 57.31% | 68.25% |
| GloVe-25 | 16,098,760 | 95.69% | 0.5139 | 6.89% | 22.29% | 17.50% | 58.74% | 70.23% |
| Last.fm ANN | 6,134,509 | 22.95% | 0.9521 | 42.93% | 86.01% | 1.32% | 6.11% | 8.53% |

Every final trace has zero malformed lines. The explicit trace flush added in
this work removed the recurring partially written final JSON record.

The distributions are not universal:

- Last.fm has an extremely small, globally hot navigation footprint. A large
  global tier is likely essential, and equal per-cluster physical partitions
  would duplicate or strand much of that footprint.
- SIFT has a moderately concentrated footprint and already shows strong gains
  after query-space grouping. It is the clearest candidate for a mixed global
  plus query-conditioned cache.
- Fashion-MNIST and GloVe touch most of the graph at high recall. Their top-node
  concentration is weaker, so cluster-specific residual working sets and
  admission selectivity matter more than a static global hot set.
- GloVe's very deep selected search increases its touched-node fraction. A fair
  cache-policy comparison must hold recall and search parameters fixed within
  the dataset, not compare raw expansion counts across datasets.

## 7. Relation to the Belady upper bound

The existing SIFT1M trace replay at capacity 10,000 gives:

| Query order | LRU hit rate | Belady-opt hit rate | Remaining gap |
|---|---:|---:|---:|
| Original | 9.21% | 24.70% | 15.49 percentage points |
| Random seed 42 | 7.91% | 23.57% | 15.66 percentage points |
| K-means grouped | 37.09% | 47.12% | 10.03 percentage points |

K-means grouping raises both the online LRU result and the offline optimum.
This is consistent with query geometry creating more short-range path overlap,
not merely making LRU better at exploiting an unchanged amount of reuse. The
remaining gap motivates query-conditioned admission and eviction rather than
only another recency/frequency heuristic.

## 8. Cache-design implications

The pilot supports a deduplicated Query-Affinity Shared Cache instead of one
independent physical cache per cluster:

1. Store each node payload once.
2. Reserve a global tier for high-frequency, early-hop, or high-affinity-entropy
   nodes.
3. Route each query softly to its nearest query prototypes.
4. Maintain decayed node-cluster affinity statistics.
5. Charge shared nodes fractionally to logical cluster quotas.
6. Allocate elastic capacity from empirical marginal hit-gain curves rather
   than query counts alone.
7. Bootstrap from base clusters only after an occupancy test like Section 4.2.

The global/conditional split must be learned per workload. Last.fm and GloVe
demonstrate that a fixed split or a quota proportional only to cluster population
is not defensible.

## 9. Limitations and next experiments

- No downloaded benchmark in this pilot provides documented production request
  timestamps. A real temporal experiment still requires an embedding workload
  derived from Last.fm 1K, MIND, KuaiRand/KuaiRec, or governed internal logs.
- Mini-batch k-means was run once at K=100 and seed 42. Publication results
  should sweep K, seeds, sample size, and update cadence.
- Current expansion traces do not encode query id, cluster id, or hop number.
  The next instrumentation step is to preserve query boundaries so that
  `A[node, query_cluster]`, affinity entropy, hot-set overlap, and per-cluster
  miss-ratio curves can be measured directly.
- The trace counts requested graph nodes. A page-aware follow-up should aggregate
  by disk sector and report physical read amplification.
- The current experiments establish measurement infrastructure and design
  evidence; they do not yet implement QASC itself.

## 10. Reproduction

Run one dataset end to end:

```bash
MAX_DEGREE=96 \
BUILD_SEARCH_LIST_SIZE=200 \
PQ_CHUNKS_OVERRIDE=16 \
INDEX_TAG=r96_l200_pq16 \
SEARCH_LISTS=50,100,200 \
SEARCH_IO_LIMIT=1000 \
experiments/cache_eval/scripts/run_spatial_locality_pilot.sh lastfm_64
```

Regenerate aggregate expansion tables and plots:

```bash
experiments/cache_eval/.venv/bin/python \
  experiments/cache_eval/scripts/summarize_spatial_locality_pilot.py \
  --pilot-root experiments/cache_eval/results/spatial_locality_pilot \
  --output-dir experiments/cache_eval/results/spatial_locality_pilot/summary
```

Primary aggregate artifacts:

```text
experiments/cache_eval/results/vector_patterns/summary/summary.md
experiments/cache_eval/results/vector_patterns/summary/base_center_occupancy_summary.csv
experiments/cache_eval/results/spatial_locality_pilot/summary/node_expansion_summary.csv
experiments/cache_eval/results/spatial_locality_pilot/summary/expansion_top_node_share.png
experiments/cache_eval/results/spatial_locality_pilot/summary/expansion_mass_coverage.png
```
