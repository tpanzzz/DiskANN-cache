# DiskANN Cache Evaluation Harness

This directory contains the first-stage harness for evaluating node-cache
policies on DiskANN disk search. It is intentionally trace-driven plus
system-level: use trace replay for fast policy sweeps, and live search for
latency/I/O validation on the actual disk index.

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
