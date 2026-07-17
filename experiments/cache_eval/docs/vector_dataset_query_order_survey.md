# Vector-search datasets and query-order semantics

## 1. Research question

This project needs more than a representative vector distribution. To evaluate
query-space locality as a cache signal, a dataset should ideally preserve:

1. realistic corpus and query embeddings;
2. query/base distribution shift when it exists in production;
3. actual query arrival order, session boundaries, or timestamps;
4. enough queries to estimate cluster transitions and drift;
5. reproducible ground truth for ANN recall.

Most standard ANN benchmarks satisfy points 1 and 5, sometimes point 2, but
almost never point 3. Their query arrays should therefore be treated as sets,
not traces, unless the dataset documentation explicitly states otherwise.

## 2. Established ANN benchmark families

### 2.1 ANN-Benchmarks

ANN-Benchmarks is a widely used reproducible CPU ANN benchmark framework. Its
HDF5 files use `train` for index/base vectors and `test` for queries, with exact
top-100 neighbors.

The generation code is important for this project:

- GloVe and several derived datasets use scikit-learn `train_test_split` with
  `random_state=1`; query order is randomized and has no arrival semantics.
- Fashion-MNIST uses the official 60,000/10,000 train/test image split; test
  order is the dataset file order, not an application query stream.
- SIFT and GIST preserve the TexMex `*_query.fvecs` order, but TexMex does not
  document that order as timestamped user traffic.
- Last.fm uses item factors as base vectors and the first 50,000 ALS user
  factors as query vectors. This is a recommendation MIPS workload, but user
  factor order is not listening-event order.

Conclusion: ANN-Benchmarks is authoritative for index speed/recall comparison,
but its query order is not evidence of production temporal locality.

Reference:

- Aumuller, Bernhardsson, and Faithfull, *ANN-Benchmarks: A Benchmarking Tool
  for Approximate Nearest Neighbor Algorithms*, Information Systems 2019,
  DOI 10.1016/j.is.2019.02.006.
- https://github.com/erikbern/ann-benchmarks

### 2.2 Big ANN Benchmarks / NeurIPS billion-scale challenge

The NeurIPS 2021/2023 Big ANN benchmarks include BIGANN/SIFT, DEEP, Microsoft
SPACEV, Yandex text-to-image, and filtered/sparse variants. They are important
for disk-resident and billion-scale evaluation.

Some source data is production-derived, especially SPACEV and Yandex
text-to-image. However, the released public query arrays are benchmark samples.
The challenge protocol does not claim that row order is the original serving
arrival order. These datasets can test query/base distribution realism and OOD
behavior, but not temporal locality without additional trace metadata.

Reference: https://big-ann-benchmarks.com/

### 2.3 VIBE

VIBE is a modern embedding benchmark with text, image, code, cross-modal,
attention-key, normalized, quantized, in-distribution, and out-of-distribution
datasets.

Its generation code shows:

- many in-distribution datasets use deterministic random splits;
- many query sets are random samples of a separate query modality;
- OOD datasets are valuable because query and base semantics differ;
- row order is generally sampler or dataset order, not an arrival trace.

VIBE is therefore highly relevant for testing whether query-cluster patterns
transfer across base/learn/query splits, but it does not by itself validate a
production query sequence.

Reference:

- Jaasaari et al., *VIBE: Vector Index Benchmark for Embeddings*, arXiv
  2505.17810, 2025.
- https://github.com/vector-index-bench/vibe

## 3. Dataset-by-dataset assessment

| Dataset | Source and generation | Scale/distance | Query construction | Does released order represent real arrival order? | Use in this project |
|---|---|---|---|---|---|
| SIFT1M | TexMex local SIFT image descriptors | 1M base, 10K query, 128d L2 | Official `sift_query.fvecs` | No documented timestamp/session semantics | Existing baseline; controlled k-means-order experiments |
| GIST1M | TexMex global image descriptors | 1M base, 1K query, 960d L2 | Official `gist_query.fvecs` | No documented temporal semantics | Useful but 3.84GB HDF5; deferred under current disk budget |
| Fashion-MNIST | Zalando grayscale product images | 60K base, 10K query, 784d L2 | Official train/test image split | No; image test-file order | Small visual-domain pilot |
| GloVe-25 | Twitter 27B token word embeddings | 1.18M base, 10K query, 25d cosine | Deterministic random train/test split | No; explicitly randomized split | Large low-dimensional semantic pilot |
| Last.fm ANN | Last.fm 360K aggregate play counts, ALS user/item factors, MIPS-to-cosine transform | 292K base, 50K query, 65d cosine | First 50K user factors | No; factor row order, not event time | Recommendation geometry and query/base-role asymmetry |
| NYTimes-256 | UCI bag-of-words, TF-IDF, random projection | 290K base, 10K query, 256d cosine | Deterministic random split | No | Text distribution pilot; not downloaded initially |
| DEEP10M/1B | CNN image descriptors | up to 1B, 96d | Random train/test-style split in common benchmark packaging | No documented temporal semantics | Disk-scale/OOD follow-up; 10M upper limit is possible but storage heavy |
| SPACEV | Microsoft Bing document/query embeddings | billion-scale | Public benchmark query samples | Production-derived, but public row order is not documented as arrival order | Strong distribution-realism follow-up |
| Yandex T2I | Image base and text query embeddings | billion-scale, 200d | Public query sample | No documented arrival semantics | Cross-modal OOD follow-up |
| COCO T2I | COCO image embeddings and caption/text embeddings | about 113K base, 10K query | Caption validation sample | No; dataset/sample order | Small OOD cross-modal pilot candidate |
| MS MARCO | Real anonymized Bing queries and web passages | 8.8M passages, more than 1M queries before embedding | Queries sampled from Bing usage logs | Query rows/IDs do not preserve timestamps or production arrival order | Real query language, but needs embedding and a separate sequence source |
| ORCAS | Bing clicked query-document pairs | about 18M pairs | Aggregated click-log pairs | Public data is not a timestamped arrival trace | Query/corpus relevance source, not sequence source |

## 4. Datasets that can preserve real sequence behavior

Standard ANN datasets are insufficient for the strongest claim. A second class
of workload should be derived from event logs.

### 4.1 AOL query log

The AOL 2006 log contains user ids, queries, timestamps, and clicked URLs. It
preserves arrival and session order, so text queries could be embedded and used
against a document corpus. However, the dataset has major privacy and ethical
issues and is not recommended as the primary public artifact.

### 4.2 Last.fm 1K listening histories

The Last.fm 1K dataset contains timestamped listening events. Item embeddings
and user-state query embeddings can produce a real sequential recommendation
retrieval workload. This is different from the ANN-Benchmarks Last.fm factor
file, which discards event order.

### 4.3 MIND news recommendation

MIND contains anonymized user impression histories with timestamps. A retrieval
workload can index news embeddings and issue user-state/query embeddings in
chronological order. It offers real sequence and session structure, though the
task is recommendation rather than generic nearest-neighbor search.

### 4.4 KuaiRand / KuaiRec and similar recommendation logs

These datasets provide timestamped interaction sequences. They can test whether
cluster transitions and drift improve a query-affinity cache. Their vector
representations must be trained or selected, so the resulting ANN workload must
publish the embedding model and preprocessing.

### 4.5 Session-track search collections

TREC Session Track and conversational-search datasets preserve within-session
query order but are usually too small for cache studies. They are useful as a
qualitative or microbenchmark workload.

## 5. Downloaded pilot datasets

Under the current limit of at most 10M base vectors and approximately 18GB free
disk space, the first pilot uses:

| Local name | HDF5 size | Base/query | Reason |
|---|---:|---:|---|
| `sift1m` | 525MB | 1,000,000 / 10,000 | Existing graph-cache baseline |
| `fashion_mnist` | 228MB | 60,000 / 10,000 | Small L2 visual dataset |
| `glove_25` | 127MB | 1,183,514 / 10,000 | Semantic cosine dataset with randomized query split |
| `lastfm_64` | 142MB | 292,385 / 50,000 | Recommendation query/base asymmetry |

The catalog and checksums are in
`experiments/cache_eval/datasets/catalog.json`. GIST1M is deferred because its
HDF5 alone is 3.84GB and index/analysis copies would consume a disproportionate
part of the remaining disk space.

## 6. Analysis required for every dataset

### 6.1 Split-level cluster distributions

Run the same `K` for base, query, and learn splits, but fit separate cluster
models. Report:

- sorted cluster-size distribution;
- cluster-size coefficient of variation and Gini coefficient;
- mass held by the largest 10% of clusters;
- mean and p95 point-to-center radius;
- sensitivity across seeds and `K`.

### 6.2 Independent-center alignment

Fit clusters independently, build the pairwise center-distance matrix, and use
Hungarian matching. Report:

- mean/median/p95 matched center distance;
- matched distance normalized by pooled cluster radius;
- Pearson and Spearman correlation of matched cluster masses;
- Jensen-Shannon divergence of matched cluster masses.

This answers whether two splits discover geometrically corresponding modes.

### 6.3 Shared-reference occupancy

Fit centers on the base split, assign query/learn vectors to those centers, and
compare occupancy distributions. This avoids instability from independently
numbered k-means solutions.

Report:

- occupancy Pearson and Spearman correlation;
- Jensen-Shannon divergence;
- fraction of query mass in low-base-mass clusters;
- query-to-nearest-base-center distance normalized by base radius.

### 6.4 Sequence-locality tests

Keep query rows in their released order and compute:

- adjacent same-cluster probability;
- cluster run-length mean/p95/max;
- conditional transition entropy;
- same-cluster probability at lags 1, 2, 5, 10, 50, 100;
- lift over permutations of the same cluster labels.

If observed metrics are indistinguishable from shuffled labels, the dataset may
have spatial patterns as a set but does not provide a meaningful locality-rich
query sequence.

### 6.5 Search-path relation

For each query cluster `z`, collect node expansion counts `A[v,z]`. Measure:

- Jaccard or weighted-Jaccard overlap between cluster hot-node sets;
- node affinity entropy across query clusters;
- fraction of expansions served by global versus cluster-specific nodes;
- cache capacity needed for 50/80/90/95% of expansion mass;
- correlation between query-center distance and search-path overlap.

These metrics determine whether a cluster-aware cache is justified and how
large its global shared tier should be.

## 7. Interpretation rules

1. Similar base/query cluster distributions do not imply temporal locality.
2. A locality lift after artificial k-means sorting proves path-overlap
   potential, not production prevalence.
3. Real sequence claims require timestamp/session evidence.
4. Base vectors are not a substitute for query demand; they are only a prior on
   corpus geometry.
5. Query ordering should always be compared with at least one fixed-seed
   shuffled baseline.
6. Report both set-level pattern similarity and sequence-level locality.

## 8. Reproducible commands

Create the analysis environment:

```bash
cd DiskANN/experiments/cache_eval
python3 -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt
```

Verify or download datasets:

```bash
cd DiskANN
experiments/cache_eval/.venv/bin/python \
  experiments/cache_eval/scripts/download_vector_datasets.py \
  --datasets fashion_mnist glove_25 lastfm_64
```

Analyze one dataset:

```bash
experiments/cache_eval/.venv/bin/python \
  experiments/cache_eval/scripts/analyze_cluster_patterns.py \
  --dataset sift1m \
  --clusters 100 \
  --seed 42
```

Run the conversion/index/search/expansion pilot:

```bash
experiments/cache_eval/scripts/run_spatial_locality_pilot.sh fashion_mnist
```

Generated analysis and experiment outputs live under
`experiments/cache_eval/results/` and are intentionally ignored by git.

## 9. Sources

- ANN-Benchmarks dataset generation and documentation:
  https://github.com/erikbern/ann-benchmarks
- ANN-Benchmarks publication: https://arxiv.org/abs/1807.05614
- VIBE benchmark: https://github.com/vector-index-bench/vibe
- Big ANN Benchmarks: https://big-ann-benchmarks.com/
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- GloVe: https://nlp.stanford.edu/projects/glove/
- TexMex SIFT/GIST corpus: http://corpus-texmex.irisa.fr/
- MS MARCO: https://microsoft.github.io/msmarco/
- MIND: https://msnews.github.io/
