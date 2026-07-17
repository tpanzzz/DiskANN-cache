# DiskANN Cache Evaluation Harness

This directory contains the first-stage harness for evaluating node-cache
policies on DiskANN disk search. It is intentionally trace-driven plus
system-level: use trace replay for fast policy sweeps, and live search for
latency/I/O validation on the actual disk index.

## Spatial Locality Research

The query-locality follow-up is documented in:

- `docs/spatial_locality_aware_cache_design.md`: query-affinity cache design,
  alternatives, capacity allocation, duplicate-node handling, and related work.
- `docs/vector_dataset_query_order_survey.md`: dataset provenance, query-order
  semantics, downloaded pilot datasets, and cross-split analysis methodology.
- `docs/spatial_locality_pilot_results_20260717.md`: measured query-order,
  cross-split cluster, high-recall search, and node-expansion results.

Create the analysis environment and verify the downloaded datasets:

```bash
python3 -m venv experiments/cache_eval/.venv
experiments/cache_eval/.venv/bin/pip install \
  -r experiments/cache_eval/requirements-analysis.txt

experiments/cache_eval/.venv/bin/python \
  experiments/cache_eval/scripts/download_vector_datasets.py \
  --datasets fashion_mnist glove_25 lastfm_64
```

Run cluster-pattern analysis:

```bash
experiments/cache_eval/.venv/bin/python \
  experiments/cache_eval/scripts/analyze_cluster_patterns.py \
  --dataset sift1m \
  --clusters 100 \
  --seed 42
```

Run the end-to-end conversion, index, recall, and node-expansion pilot for one
catalog dataset:

```bash
experiments/cache_eval/scripts/run_spatial_locality_pilot.sh fashion_mnist
```

The pilot accepts environment overrides for graph, PQ, search, and output-tag
parameters. For example:

```bash
MAX_DEGREE=96 \
BUILD_SEARCH_LIST_SIZE=200 \
PQ_CHUNKS_OVERRIDE=16 \
INDEX_TAG=r96_l200_pq16 \
SEARCH_LISTS=50,100,200 \
SEARCH_IO_LIMIT=1000 \
experiments/cache_eval/scripts/run_spatial_locality_pilot.sh lastfm_64
```

Aggregate completed expansion summaries:

```bash
experiments/cache_eval/.venv/bin/python \
  experiments/cache_eval/scripts/summarize_spatial_locality_pilot.py \
  --pilot-root experiments/cache_eval/results/spatial_locality_pilot \
  --output-dir experiments/cache_eval/results/spatial_locality_pilot/summary
```

Dataset metadata, expected file sizes, and checksums are recorded in
`datasets/catalog.json`. Generated analysis and index outputs are written under
`experiments/cache_eval/results/` and are ignored by git.

## Query Workloads

Generate a reproducible random order:

```bash
cargo run -q -p diskann-tools --bin query_workload --release -- random-order \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --data_type float \
  --seed 42 \
  --order_output experiments/cache_eval/sift1m/workloads/random_seed42.order.csv
```

Generate a k-means locality order. K-means and all greedy ordering steps are
deterministic for the fixed seed and thread count:

```bash
cargo run -q -p diskann-tools --bin query_workload --release -- kmeans-order \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --clusters 100 \
  --seed 42 \
  --max_reps 20 \
  --num_threads 1 \
  --order_output experiments/cache_eval/sift1m/workloads/kmeans_k100_seed42.order.csv \
  --assignments_output experiments/cache_eval/sift1m/workloads/kmeans_k100_seed42.assignments.csv
```

Apply an order to both query vectors and ground truth:

```bash
cargo run -q -p diskann-tools --bin query_workload --release -- apply-order \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --ground_truth_file experiments/cache_eval/sift1m/groundtruth.bin \
  --data_type float \
  --order_file experiments/cache_eval/sift1m/workloads/kmeans_k100_seed42.order.csv \
  --query_output experiments/cache_eval/sift1m/workloads/query.kmeans_k100_seed42.fbin \
  --ground_truth_output experiments/cache_eval/sift1m/workloads/groundtruth.kmeans_k100_seed42.bin
```

Order files are one-column CSV files with a `query_id` header and a complete,
duplicate-free permutation of `0..nqueries`. They can also be edited or written
by hand. `apply-order` supports `float`, `int8`, `uint8`, and `fp16` query
matrices and preserves whether the input truthset contains distances.

## Convert SIFT HDF5

Inspect the local HDF5 keys first because mirrors may differ:

```bash
python experiments/cache_eval/hdf5_to_diskann_bin.py \
  --input ../datasets/sift-128-euclidean.hdf5 \
  --output-dir experiments/cache_eval/sift1m \
  --inspect
```

Convert the common `train/test/neighbors/distances` layout:

```bash
python experiments/cache_eval/hdf5_to_diskann_bin.py \
  --input ../datasets/sift-128-euclidean.hdf5 \
  --output-dir experiments/cache_eval/sift1m
```

The converter writes:

- `base.fbin`: `f32` base vectors in DiskANN binary matrix format.
- `query.fbin`: `f32` query vectors in DiskANN binary matrix format.
- `groundtruth.bin`: `u32` neighbor IDs plus optional `f32` distances.

## Build Index

```bash
cargo run -p diskann-tools --bin build_disk_index --release -- \
  --data_type float \
  --dist_fn l2 \
  --data_path experiments/cache_eval/sift1m/base.fbin \
  --index_path_prefix experiments/cache_eval/sift1m/sift_disk \
  --dimensions 128 \
  --full_precision_dimensions 128 \
  --max_degree 64 \
  --build_search_list_size 100 \
  --pq_chunks 16 \
  --index_build_ram_limit_gb 8
```

## Live Search Baselines

No cache:

```bash
cargo run -p diskann-tools --bin search_disk_index --release -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix experiments/cache_eval/sift1m/sift_disk \
  --result_output_prefix experiments/cache_eval/sift1m/results/no_cache \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --ground_truth_file experiments/cache_eval/sift1m/groundtruth.bin \
  --search_list 20,50,100,200 \
  --beam_width 4 \
  --recall_at 10 \
  --num_nodes_to_cache 0
```

Original DiskANN static medoid-BFS cache:

```bash
cargo run -p diskann-tools --bin search_disk_index --release -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix experiments/cache_eval/sift1m/sift_disk \
  --result_output_prefix experiments/cache_eval/sift1m/results/static_bfs_10000 \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --ground_truth_file experiments/cache_eval/sift1m/groundtruth.bin \
  --search_list 20,50,100,200 \
  --beam_width 4 \
  --recall_at 10 \
  --num_nodes_to_cache 10000
```

Dynamic LRU with a BFS warm start:

```bash
cargo run -p diskann-tools --bin search_disk_index --release -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix experiments/cache_eval/sift1m/sift_disk \
  --result_output_prefix experiments/cache_eval/sift1m/results/lru_warm_10000 \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --ground_truth_file experiments/cache_eval/sift1m/groundtruth.bin \
  --search_list 20,50,100,200 \
  --beam_width 4 \
  --recall_at 10 \
  --cache_policy lru \
  --cache_capacity 10000 \
  --cache_warmup_nodes 5000
```

## Trace Replay

Collect an access trace during live search by setting `DISKANN_CACHE_TRACE_PATH`.
Each line is JSON with `stage`, `vertex_id`, `cache_hit`, and `cache_source`.

```bash
mkdir -p experiments/cache_eval/traces
DISKANN_CACHE_TRACE_PATH=experiments/cache_eval/traces/sift_accesses.jsonl \
cargo run -p diskann-tools --bin search_disk_index --release -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix experiments/cache_eval/sift1m/sift_disk \
  --result_output_prefix experiments/cache_eval/sift1m/results/trace_static_bfs \
  --query_file experiments/cache_eval/sift1m/query.fbin \
  --ground_truth_file experiments/cache_eval/sift1m/groundtruth.bin \
  --search_list 100 \
  --beam_width 4 \
  --recall_at 10 \
  --num_nodes_to_cache 10000
```

Replay a text trace where column 0 is a vertex id:

```bash
cargo run -p diskann-tools --bin cache_trace_replay --release -- \
  --trace experiments/cache_eval/traces/sift_accesses.jsonl \
  --capacities 1000,5000,10000,20000 \
  --policies no_cache,fifo,lru,lfu,tiny_lfu,random,belady_opt \
  --output experiments/cache_eval/results/trace_replay.csv
```

The replay tool also accepts JSONL lines with a numeric `vertex_id`, `id`, or
`vertex` field.
