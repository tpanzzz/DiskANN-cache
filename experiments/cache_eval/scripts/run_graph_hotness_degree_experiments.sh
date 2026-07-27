#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${ROOT_DIR}/experiments/cache_eval/.venv/bin/python"
SEARCH_BIN="${ROOT_DIR}/target/release/search_disk_index"
OUTPUT_ROOT="${ROOT_DIR}/experiments/cache_eval/results/graph_hotness_degree"
SEED="${SEED:-42}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.01}"
NUM_THREADS="${NUM_THREADS:-8}"
DATASET="${1:-all}"

run_search() {
  local source_name="$1"
  local query_file="$2"
  local groundtruth_file="$3"
  local recall_at="$4"
  local run_dir="$5"
  local fail_threshold="$6"

  local counts_file="${run_dir}/${source_name}.node_expansion_counts.csv"
  local result_prefix="${run_dir}/${source_name}"
  local log_file="${run_dir}/${source_name}.search.log"
  local navigation_args=()
  if [[ -n "${FULL_PRECISION_FILE}" ]]; then
    navigation_args=(--full_precision_vector_file "${FULL_PRECISION_FILE}")
  fi

  DISKANN_NODE_ACCESS_COUNTS_PATH="${counts_file}" \
    DISKANN_NODE_ACCESS_NUM_NODES="${NUM_NODES}" \
    /usr/bin/time -v "${SEARCH_BIN}" \
      --data_type float \
      --dist_fn "${METRIC}" \
      --index_path_prefix "${INDEX_PREFIX}" \
      "${navigation_args[@]}" \
      --result_output_prefix "${result_prefix}" \
      --query_file "${query_file}" \
      --ground_truth_file "${groundtruth_file}" \
      --search_list "${SEARCH_LIST}" \
      --beam_width 4 \
      --recall_at "${recall_at}" \
      --search_io_limit "${SEARCH_IO_LIMIT}" \
      --num_threads "${NUM_THREADS}" \
      --num_nodes_to_cache 0 \
      --fail_if_recall_below "${fail_threshold}" \
      2>&1 | tee "${log_file}"
}

run_dataset() {
  local name="$1"
  local display_name dimensions base_file test_query test_groundtruth
  FULL_PRECISION_FILE=""

  case "${name}" in
    sift1m)
      display_name="SIFT1M"
      dimensions=128
      NUM_NODES=1000000
      base_file="${ROOT_DIR}/experiments/cache_eval/sift1m/base.fbin"
      test_query="${ROOT_DIR}/experiments/cache_eval/sift1m/query.fbin"
      test_groundtruth="${ROOT_DIR}/experiments/cache_eval/sift1m/groundtruth.bin"
      INDEX_PREFIX="${ROOT_DIR}/experiments/cache_eval/sift1m/sift_disk"
      METRIC="l2"
      SEARCH_LIST=100
      SEARCH_IO_LIMIT=1000
      PQ_CHUNKS=16
      MAX_DEGREE=64
      BUILD_SEARCH_LIST=100
      ;;
    fashion_mnist)
      display_name="Fashion-MNIST"
      dimensions=784
      NUM_NODES=60000
      base_file="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/fashion_mnist/data/base.fbin"
      test_query="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/fashion_mnist/data/query.fbin"
      test_groundtruth="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/fashion_mnist/data/groundtruth.bin"
      INDEX_PREFIX="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/fashion_mnist/index/fashion_mnist"
      METRIC="l2"
      SEARCH_LIST=50
      SEARCH_IO_LIMIT=1000
      PQ_CHUNKS=49
      MAX_DEGREE=64
      BUILD_SEARCH_LIST=100
      ;;
    glove_25)
      display_name="GloVe-25 normalized"
      dimensions=25
      NUM_NODES=1183514
      base_file="${ROOT_DIR}/experiments/cache_eval/results/glove_pq_sweep/data/base.normalized.fbin"
      test_query="${ROOT_DIR}/experiments/cache_eval/results/glove_pq_sweep/data/query.normalized.fbin"
      test_groundtruth="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/glove_25/data/groundtruth.bin"
      INDEX_PREFIX="${ROOT_DIR}/experiments/cache_eval/results/glove_pq_sweep/index/normalized/pq16/glove_normalized_pq16"
      METRIC="cosinenormalized"
      SEARCH_LIST=20
      SEARCH_IO_LIMIT=2000
      PQ_CHUNKS=16
      MAX_DEGREE=96
      BUILD_SEARCH_LIST=200
      ;;
    lastfm_64)
      display_name="Last.fm ANN"
      dimensions=65
      NUM_NODES=292385
      base_file="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/lastfm_64/data/base.fbin"
      test_query="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/lastfm_64/data/query.fbin"
      test_groundtruth="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/lastfm_64/data/groundtruth.bin"
      INDEX_PREFIX="${ROOT_DIR}/experiments/cache_eval/results/graph_hotness_degree/lastfm_64/index/lastfm_64_r128_l400"
      METRIC="cosine"
      SEARCH_LIST=20
      SEARCH_IO_LIMIT=1000
      PQ_CHUNKS=0
      MAX_DEGREE=128
      BUILD_SEARCH_LIST=400
      FULL_PRECISION_FILE="${base_file}"
      ;;
    *)
      echo "unknown dataset: ${name}" >&2
      return 2
      ;;
  esac

  local dataset_dir="${OUTPUT_ROOT}/${name}"
  local workload_dir="${dataset_dir}/workloads"
  local run_dir="${dataset_dir}/search"
  local analysis_dir="${dataset_dir}/analysis"
  local index_file="${INDEX_PREFIX}_disk.index"
  mkdir -p "${workload_dir}" "${run_dir}" "${analysis_dir}"

  "${PYTHON}" "${ROOT_DIR}/experiments/cache_eval/scripts/prepare_base_query_workloads.py" \
    --base-file "${base_file}" \
    --output-dir "${workload_dir}" \
    --sample-fraction "${SAMPLE_FRACTION}" \
    --seed "${SEED}"

  local all_groundtruth="${workload_dir}/base_all.identity_groundtruth.bin"
  local sample_query="${workload_dir}/base_sample_${SAMPLE_FRACTION}_seed${SEED}.fbin"
  local sample_groundtruth="${workload_dir}/base_sample_${SAMPLE_FRACTION}_seed${SEED}.identity_groundtruth.bin"

  run_search test_queries "${test_query}" "${test_groundtruth}" 10 "${run_dir}" 95
  run_search all_base_queries "${base_file}" "${all_groundtruth}" 1 "${run_dir}" 0
  run_search sampled_base_queries "${sample_query}" "${sample_groundtruth}" 1 "${run_dir}" 0

  local config_file="${dataset_dir}/search_config.json"
  "${PYTHON}" - \
    "${config_file}" "${name}" "${display_name}" "${METRIC}" "${SEARCH_LIST}" \
    "${SEARCH_IO_LIMIT}" "${PQ_CHUNKS}" "${SEED}" "${SAMPLE_FRACTION}" \
    "${NUM_THREADS}" "${MAX_DEGREE}" "${BUILD_SEARCH_LIST}" \
    "${FULL_PRECISION_FILE}" \
    "$(git -C "${ROOT_DIR}" rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    dataset,
    display_name,
    metric,
    search_list,
    io_limit,
    pq_chunks,
    seed,
    sample_fraction,
    num_threads,
    max_degree,
    build_search_list,
    full_precision_file,
    git_commit,
) = sys.argv[1:]
value = {
    "dataset": dataset,
    "display_name": display_name,
    "metric": metric,
    "search_list": int(search_list),
    "beam_width": 4,
    "search_io_limit": int(io_limit),
    "num_threads": int(num_threads),
    "max_degree": int(max_degree),
    "build_search_list": int(build_search_list),
    "cache": "disabled",
    "pq_chunks": int(pq_chunks),
    "navigation": "full_precision" if full_precision_file else "pq",
    "full_precision_vector_file": full_precision_file or None,
    "test_quality_gate": "Recall@10 >= 95%",
    "base_quality_gate": "distance-equivalent self Recall@1 >= 95%",
    "sample_seed": int(seed),
    "sample_fraction": float(sample_fraction),
    "git_commit": git_commit,
}
Path(output).write_text(json.dumps(value, indent=2) + "\n")
PY

  local self_quality_file="${dataset_dir}/self_query_quality.json"
  "${PYTHON}" "${ROOT_DIR}/experiments/cache_eval/scripts/validate_self_query_results.py" \
    --base-file "${base_file}" \
    --all-result "${run_dir}/all_base_queries_${SEARCH_LIST}_idx_uint32.bin" \
    --sample-result "${run_dir}/sampled_base_queries_${SEARCH_LIST}_idx_uint32.bin" \
    --sample-ids "${workload_dir}/base_sample_${SAMPLE_FRACTION}_seed${SEED}.ids.csv" \
    --metric "${METRIC}" \
    --distance-tolerance 1e-6 \
    --required-recall 0.95 \
    --output "${self_quality_file}"

  "${PYTHON}" "${ROOT_DIR}/experiments/cache_eval/scripts/analyze_graph_hotness_degree.py" \
    --dataset "${display_name}" \
    --index-file "${index_file}" \
    --dimensions "${dimensions}" \
    --test-counts "${run_dir}/test_queries.node_expansion_counts.csv" \
    --base-counts "${run_dir}/all_base_queries.node_expansion_counts.csv" \
    --sample-counts "${run_dir}/sampled_base_queries.node_expansion_counts.csv" \
    --search-config "${config_file}" \
    --self-quality "${self_quality_file}" \
    --input-file "${base_file}" \
    --input-file "${test_query}" \
    --input-file "${test_groundtruth}" \
    --input-file "${workload_dir}/workload_manifest.json" \
    --output-dir "${analysis_dir}"
}

cargo build -q -p diskann-tools --bin search_disk_index --release

if [[ "${DATASET}" == "all" ]]; then
  run_dataset sift1m
  run_dataset fashion_mnist
  run_dataset glove_25
  run_dataset lastfm_64
else
  run_dataset "${DATASET}"
fi
