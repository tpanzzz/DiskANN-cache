# GloVe cosine, PQ, and full-precision navigation results, 2026-07-17

## 1. Question

The previous GloVe-25 pilot used five PQ chunks and required `L=1600` to reach
Recall@10=95.94%. This follow-up evaluates:

- PQ chunks 5, 10, 16, and 25;
- original unnormalized vectors versus unit-normalized vectors;
- an exact in-memory full-precision navigation path that does not load or use PQ;
- whether the previous cosine experiment performed the transformation required
  by the PQ search implementation.

## 2. Was cosine converted correctly in the previous experiment?

Only partially.

The ANN-Benchmarks HDF5 has `distance=angular`, but its vectors are not unit
normalized. Measured norms were:

| Split | Minimum norm | Maximum norm |
|---|---:|---:|
| Base | 0.0753422 | 17.0807 |
| Query | 1.57327 | 17.8603 |

The previous index build and search both passed `--dist_fn cosine`. Therefore:

- graph construction used the repository's true unnormalized cosine distance;
- full-precision distance calculation and final reranking used true cosine;
- the graph was not accidentally constructed with raw L2 distance.

However, `quantizer_preprocess.rs` explicitly maps `Metric::Cosine` and
`Metric::CosineNormalized` to `SquaredL2` for PQ candidate distances. With raw,
unnormalized GloVe vectors, raw L2 ordering is not equivalent to cosine ordering.
The previous experiment did not normalize the vectors before PQ encoding, so
the graph target metric and the PQ navigation metric were inconsistent.

For unit vectors `x` and `q`:

```text
||x - q||^2 = 2 - 2 * cosine_similarity(x, q)
```

Unit normalization therefore makes squared-L2 PQ ordering equivalent to cosine
ordering. The corrected normalized experiment uses `Metric::CosineNormalized`
for graph construction and exact distance calculations, while the existing
L2-based PQ navigation becomes mathematically aligned with the target metric.

An exact check over 20 distributed query ids found 100% top-10 overlap between
normalized inner-product search and the released angular ground truth.

## 3. Experimental configuration

- dataset: GloVe Twitter 25d from ANN-Benchmarks;
- base vectors: 1,183,514;
- queries: 10,000, complete released test split;
- graph degree R: 96;
- graph build L: 200;
- search threads: 1;
- beam width: 4;
- search I/O limit: 2000;
- recall target: Recall@10 >= 95%;
- node cache: disabled;
- tested L values: 10, 20, 30, 40, 50, 75, 100, 150, 200, 300,
  400, 600, 800, 1200, and 1600 as needed.

One deterministic raw-cosine graph and one deterministic normalized-cosine
graph were built. Within each representation, every PQ variant uses a hard link
to the same `_disk.index`; only its PQ pivots and compressed codes differ. This
prevents graph variation from contaminating the chunk comparison.

The full-precision mode loads the complete f32 base matrix into memory and uses
the configured exact metric for every graph-navigation comparison. It does not
open PQ pivot or compressed-code files and does not allocate PQ scratch. The
disk graph still supplies adjacency lists and full node records.

## 4. Minimum tested L reaching 95% recall

| Representation | Navigation | L | Recall@10 | QPS | Mean latency (us) | P99.9 (us) | Mean I/O | Navigation memory | Max RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | PQ5 | 1600 | 95.94% | 23.44 | 42,655.31 | 47,944 | 1609.87 | 5.64 MiB | 20.73 MiB |
| Raw | PQ10 | 200 | 95.27% | 175.52 | 5,696.00 | 7,633 | 213.12 | 11.29 MiB | 25.87 MiB |
| Raw | PQ16 | 150 | 95.68% | 225.50 | 4,433.19 | 6,559 | 163.35 | 18.06 MiB | 32.00 MiB |
| Raw | PQ25 | 150 | 95.93% | 222.49 | 4,492.78 | 6,859 | 163.19 | 28.22 MiB | 42.60 MiB |
| Raw | Full precision | 10 | 95.25% | 1191.78 | 838.01 | 2,093 | 25.17 | 112.87 MiB | 126.79 MiB |
| Normalized | PQ5 | 600 | 96.37% | 62.36 | 16,035.12 | 18,019 | 609.71 | 5.64 MiB | 20.12 MiB |
| Normalized | PQ10 | 40 | 95.20% | 665.60 | 1,501.28 | 2,296 | 52.75 | 11.29 MiB | 25.98 MiB |
| Normalized | PQ16 | 20 | 96.79% | 1007.12 | 991.87 | 1,651 | 33.59 | 18.06 MiB | 32.29 MiB |
| Normalized | PQ25 | 20 | 97.98% | 978.91 | 1,020.48 | 1,730 | 33.50 | 28.22 MiB | 42.50 MiB |
| Normalized | Full precision | 10 | 95.21% | 1289.83 | 774.25 | 1,676 | 24.79 | 112.87 MiB | 126.79 MiB |

These are the minimum values in the tested L grid, not a claim that no integer
between two tested values could satisfy the target.

## 5. Main findings

### 5.1 The graph itself was not the reason L had to reach 1600

On the same raw-cosine graph, replacing PQ5 navigation with exact in-memory
cosine navigation changes the required L from 1600 to 10:

```text
mean I/O:     1609.87 -> 25.17, 63.96x reduction
mean latency: 42.66ms -> 0.84ms, 50.90x reduction
QPS:          23.44   -> 1191.78, 50.84x increase
```

This establishes that the graph supports high-recall search at very low L. The
previous search budget was compensating for poor PQ navigation.

### 5.2 More chunks help raw vectors, but do not fix metric mismatch

Increasing raw-vector PQ from 5 to 10 chunks reduces the target L from 1600 to
200. PQ16 and PQ25 reduce it to 150, but PQ25 does not improve the tested target
L over PQ16 despite using 56% more code memory.

Even PQ25 stores one quantized byte per original dimension; it still evaluates
an approximation of raw L2 rather than cosine. Finer quantization reduces error,
but cannot make two different metrics identical.

### 5.3 Normalization plus enough chunks gives most of full precision's benefit

With normalized vectors:

- PQ10 reaches the target at L=40 using 11.29 MiB of codes;
- PQ16 reaches 96.79% at L=20 using 18.06 MiB;
- PQ25 also needs L=20 and is slightly slower at the selected point;
- full precision reaches the target at L=10 using 112.87 MiB.

Normalized PQ16 is the strongest practical operating point in this sweep. It
uses 6.25x less navigation-vector memory than full precision, while achieving:

```text
Recall@10: 96.79%
mean I/O:  33.59 versus full precision 24.79
latency:   0.99ms versus full precision 0.77ms
QPS:       1007 versus full precision 1290
```

### 5.4 Five chunks remain too coarse even after normalization

Normalization improves PQ5 from L=1600 to L=600, confirming that metric
alignment matters. It still remains far behind PQ10/PQ16 because each chunk
must represent five dimensions with one byte. Metric alignment and sufficient
quantization granularity are both necessary.

## 6. Recommendation

For GloVe-25 and similar cosine workloads:

1. Unit-normalize base and query vectors offline.
2. Build and search with `cosinenormalized`, or use L2 consistently on the unit
   vectors when the implementation permits.
3. Use PQ16 as the default tested operating point for GloVe-25.
4. Use full-precision in-memory navigation when an additional approximately
   113 MiB for 1.18M x 25 f32 vectors is acceptable and minimum I/O is preferred.
5. Do not use raw unnormalized cosine vectors with the current L2 PQ navigation
   without explicitly accepting the metric mismatch.

The search CLI now emits a warning when `Metric::Cosine` is combined with PQ
navigation. Passing `--full_precision_vector_file` selects exact navigation and
requires only the disk graph; PQ files can be absent.

## 7. Performance caveat

Recall and logical I/O are the strongest comparison metrics here. Runs were
executed sequentially on one machine, so operating-system page-cache state can
affect QPS and latency. A publication-quality latency comparison should repeat
each selected point, randomize run order, report warm/cold page-cache state, and
include confidence intervals.

## 8. Artifacts

```text
experiments/cache_eval/results/glove_pq_sweep/glove_pq_sweep_summary.csv
experiments/cache_eval/results/glove_pq_sweep/summary/selected_target_recall.csv
experiments/cache_eval/results/glove_pq_sweep/summary/recall_vs_search_l.png
experiments/cache_eval/results/glove_pq_sweep/summary/target_recall_tradeoffs.png
```

Reproduce the sweep:

```bash
experiments/cache_eval/scripts/run_glove_pq_sweep.sh
```

Regenerate the summary:

```bash
experiments/cache_eval/.venv/bin/python \
  experiments/cache_eval/scripts/summarize_glove_pq_sweep.py \
  --input experiments/cache_eval/results/glove_pq_sweep/glove_pq_sweep_summary.csv \
  --output-dir experiments/cache_eval/results/glove_pq_sweep/summary
```
