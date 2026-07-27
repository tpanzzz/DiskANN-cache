#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${ROOT_DIR}/experiments/cache_eval/.venv/bin/python"
SEARCH_BIN="${ROOT_DIR}/target/release/search_disk_index"
HOTNESS_ROOT="${ROOT_DIR}/experiments/cache_eval/results/graph_hotness_degree"
OUTPUT_ROOT="${ROOT_DIR}/experiments/cache_eval/results/static_cache_sources"
REPETITIONS="${REPETITIONS:-3}"
DATASET="${1:-all}"
MANIFEST="${OUTPUT_ROOT}/run_manifest.csv"

mkdir -p "${OUTPUT_ROOT}"
printf '%s\n' 'dataset,source,repetition,cache_capacity_nodes,queries,search_list,beam_width,log_file' > "${MANIFEST}"

configure_dataset() {
  local name="$1"
  FULL_PRECISION_FILE=""
  case "${name}" in
    sift1m)
      DISPLAY_NAME="SIFT1M"; DIMENSIONS=128; NUM_NODES=1000000
      QUERY_FILE="${ROOT_DIR}/experiments/cache_eval/sift1m/query.fbin"
      GROUNDTRUTH_FILE="${ROOT_DIR}/experiments/cache_eval/sift1m/groundtruth.bin"
      INDEX_PREFIX="${ROOT_DIR}/experiments/cache_eval/sift1m/sift_disk"
      METRIC="l2"; SEARCH_LIST=100; SEARCH_IO_LIMIT=1000; QUERIES=10000 ;;
    fashion_mnist)
      DISPLAY_NAME="Fashion-MNIST"; DIMENSIONS=784; NUM_NODES=60000
      QUERY_FILE="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/fashion_mnist/data/query.fbin"
      GROUNDTRUTH_FILE="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/fashion_mnist/data/groundtruth.bin"
      INDEX_PREFIX="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/fashion_mnist/index/fashion_mnist"
      METRIC="l2"; SEARCH_LIST=50; SEARCH_IO_LIMIT=1000; QUERIES=10000 ;;
    glove_25)
      DISPLAY_NAME="GloVe-25 normalized"; DIMENSIONS=25; NUM_NODES=1183514
      QUERY_FILE="${ROOT_DIR}/experiments/cache_eval/results/glove_pq_sweep/data/query.normalized.fbin"
      GROUNDTRUTH_FILE="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/glove_25/data/groundtruth.bin"
      INDEX_PREFIX="${ROOT_DIR}/experiments/cache_eval/results/glove_pq_sweep/index/normalized/pq16/glove_normalized_pq16"
      METRIC="cosinenormalized"; SEARCH_LIST=20; SEARCH_IO_LIMIT=2000; QUERIES=10000 ;;
    lastfm_64)
      DISPLAY_NAME="Last.fm ANN"; DIMENSIONS=65; NUM_NODES=292385
      QUERY_FILE="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/lastfm_64/data/query.fbin"
      GROUNDTRUTH_FILE="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/lastfm_64/data/groundtruth.bin"
      INDEX_PREFIX="${ROOT_DIR}/experiments/cache_eval/results/graph_hotness_degree/lastfm_64/index/lastfm_64_r128_l400"
      METRIC="cosine"; SEARCH_LIST=20; SEARCH_IO_LIMIT=1000; QUERIES=50000
      FULL_PRECISION_FILE="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/lastfm_64/data/base.fbin" ;;
    *) echo "unknown dataset: ${name}" >&2; return 2 ;;
  esac
}

prepare_sources() {
  local name="$1"
  local dataset_hotness="${HOTNESS_ROOT}/${name}"
  local source_dir="${OUTPUT_ROOT}/${name}/sources"
  mkdir -p "${source_dir}"
  "${PYTHON}" "${ROOT_DIR}/experiments/cache_eval/scripts/analyze_static_cache_sources.py" \
    --dataset "${DISPLAY_NAME}" \
    --index-file "${INDEX_PREFIX}_disk.index" \
    --dimensions "${DIMENSIONS}" \
    --test-counts "${dataset_hotness}/search/test_queries.node_expansion_counts.csv" \
    --base-counts "${dataset_hotness}/search/all_base_queries.node_expansion_counts.csv" \
    --sample-counts "${dataset_hotness}/search/sampled_base_queries.node_expansion_counts.csv" \
    --cache-fraction 0.01 \
    --output-dir "${source_dir}"
}

run_one() {
  local name="$1" source="$2" repetition="$3"
  local dataset_dir="${OUTPUT_ROOT}/${name}"
  local source_file="${dataset_dir}/sources/${source}.nodes.csv"
  local run_dir="${dataset_dir}/benchmarks"
  local prefix="${run_dir}/${source}.rep${repetition}"
  local log_file="${prefix}.log"
  local capacity
  capacity="$(( (NUM_NODES + 99) / 100 ))"
  mkdir -p "${run_dir}"
  local navigation_args=()
  if [[ -n "${FULL_PRECISION_FILE}" ]]; then
    navigation_args=(--full_precision_vector_file "${FULL_PRECISION_FILE}")
  fi
  "${SEARCH_BIN}" \
    --data_type float \
    --dist_fn "${METRIC}" \
    --index_path_prefix "${INDEX_PREFIX}" \
    "${navigation_args[@]}" \
    --result_output_prefix "${prefix}" \
    --query_file "${QUERY_FILE}" \
    --ground_truth_file "${GROUNDTRUTH_FILE}" \
    --search_list "${SEARCH_LIST}" \
    --beam_width 4 \
    --recall_at 10 \
    --search_io_limit "${SEARCH_IO_LIMIT}" \
    --num_threads 1 \
    --static_cache_nodes_file "${source_file}" \
    --fail_if_recall_below 95 \
    2>&1 | tee "${log_file}"
  printf '%s\n' "${name},${source},${repetition},${capacity},${QUERIES},${SEARCH_LIST},4,${log_file}" >> "${MANIFEST}"
}

run_dataset() {
  local name="$1"
  configure_dataset "${name}"
  prepare_sources "${name}"
  local sources=(bfs all_base_hot sample_base_hot test_hot)
  local repetition offset index source
  for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
    offset=$(( (repetition - 1) % ${#sources[@]} ))
    for ((index = 0; index < ${#sources[@]}; index++)); do
      source="${sources[$(( (index + offset) % ${#sources[@]} ))]}"
      run_one "${name}" "${source}" "${repetition}"
    done
  done
}

cargo build -q -p diskann-tools --bin search_disk_index --release
if [[ "${DATASET}" == "all" ]]; then
  for dataset in sift1m fashion_mnist glove_25 lastfm_64; do
    run_dataset "${dataset}"
  done
else
  run_dataset "${DATASET}"
fi

"${PYTHON}" "${ROOT_DIR}/experiments/cache_eval/scripts/summarize_static_cache_source_benchmarks.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}/summary"
