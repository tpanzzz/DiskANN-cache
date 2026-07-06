#!/usr/bin/env bash
set -euo pipefail

# Run from the DiskANN repository root.
#
# This sweep keeps the nominal cache-node budget fixed across single-tier and
# hybrid policies. Baselines use TOTAL_CACHE_NODES nodes. Hybrid policies use
# STATIC:DYNAMIC splits whose sum must equal TOTAL_CACHE_NODES.

DATASET_DIR="${DATASET_DIR:-experiments/cache_eval/sift1m}"
INDEX_PREFIX="${INDEX_PREFIX:-${DATASET_DIR}/sift_disk}"
RESULT_ROOT="${RESULT_ROOT:-${DATASET_DIR}/results/concurrent_cache_fair_split}"
THREADS_LIST="${THREADS_LIST:-1 2 4 8 16 32}"
WORKLOADS_LIST="${WORKLOADS_LIST:-original random_seed42 kmeans_k100_seed42}"

TOTAL_CACHE_NODES="${TOTAL_CACHE_NODES:-10000}"
SPLITS="${SPLITS:-2500:7500 5000:5000 7500:2500}"
SHARDS="${SHARDS:-64}"
SEARCH_LIST="${SEARCH_LIST:-100}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
RECALL_AT="${RECALL_AT:-10}"
SEARCH_IO_LIMIT="${SEARCH_IO_LIMIT:-1000}"
DATA_TYPE="${DATA_TYPE:-float}"
DIST_FN="${DIST_FN:-l2}"
TRACE_CACHE_ACCESS="${TRACE_CACHE_ACCESS:-0}"

if [[ "${TOTAL_CACHE_NODES}" -le 0 ]]; then
  printf 'TOTAL_CACHE_NODES must be greater than 0\n' >&2
  exit 1
fi

mkdir -p "${RESULT_ROOT}/logs"
MANIFEST="${RESULT_ROOT}/run_manifest.csv"
MANIFEST_HEADER='workload,policy,backend,shards,admission,threads,cache_capacity_nodes,static_cache_nodes,query_file,ground_truth_file,search_list,beam_width,recall_at,search_io_limit,result_prefix,log_file,trace_file'

if [[ ! -f "${MANIFEST}" ]]; then
  printf '%s\n' "${MANIFEST_HEADER}" > "${MANIFEST}"
else
  read -r EXISTING_MANIFEST_HEADER < "${MANIFEST}"
  if [[ "${EXISTING_MANIFEST_HEADER}" != "${MANIFEST_HEADER}" ]]; then
    printf 'manifest header mismatch in %s\n' "${MANIFEST}" >&2
    printf 'existing: %s\n' "${EXISTING_MANIFEST_HEADER}" >&2
    printf 'expected: %s\n' "${MANIFEST_HEADER}" >&2
    printf 'Use a fresh RESULT_ROOT or move the existing manifest before running this script.\n' >&2
    exit 1
  fi
fi

query_file_for_workload() {
  case "$1" in
    original) printf '%s\n' "${DATASET_DIR}/query.fbin" ;;
    random_seed42) printf '%s\n' "${DATASET_DIR}/workloads/query.random_seed42.fbin" ;;
    kmeans_k100_seed42) printf '%s\n' "${DATASET_DIR}/workloads/query.kmeans_k100_seed42.fbin" ;;
    *) printf 'unknown workload: %s\n' "$1" >&2; return 1 ;;
  esac
}

truth_file_for_workload() {
  case "$1" in
    original) printf '%s\n' "${DATASET_DIR}/groundtruth.bin" ;;
    random_seed42) printf '%s\n' "${DATASET_DIR}/workloads/groundtruth.random_seed42.bin" ;;
    kmeans_k100_seed42) printf '%s\n' "${DATASET_DIR}/workloads/groundtruth.kmeans_k100_seed42.bin" ;;
    *) printf 'unknown workload: %s\n' "$1" >&2; return 1 ;;
  esac
}

validate_split() {
  local split="$1"
  local static_nodes="${split%%:*}"
  local dynamic_nodes="${split##*:}"

  if [[ "${split}" != *:* || -z "${static_nodes}" || -z "${dynamic_nodes}" ]]; then
    printf 'invalid split %s; expected STATIC:DYNAMIC\n' "${split}" >&2
    return 1
  fi
  if [[ "${static_nodes}" -le 0 || "${dynamic_nodes}" -le 0 ]]; then
    printf 'invalid split %s; both sides must be greater than 0\n' "${split}" >&2
    return 1
  fi
  if (( static_nodes + dynamic_nodes != TOTAL_CACHE_NODES )); then
    printf 'invalid split %s; %s + %s != TOTAL_CACHE_NODES=%s\n' \
      "${split}" "${static_nodes}" "${dynamic_nodes}" "${TOTAL_CACHE_NODES}" >&2
    return 1
  fi
}

for split in ${SPLITS}; do
  validate_split "${split}"
done

run_search() {
  local workload="$1"
  local policy_label="$2"
  local backend="$3"
  local shards="$4"
  local admission="$5"
  local threads="$6"
  local cache_capacity="$7"
  local static_nodes="$8"
  shift 8

  local query_file
  local truth_file
  query_file="$(query_file_for_workload "${workload}")"
  truth_file="$(truth_file_for_workload "${workload}")"

  local safe_policy="${policy_label//[^A-Za-z0-9_]/_}"
  local result_prefix="${RESULT_ROOT}/${workload}_${safe_policy}_${threads}t"
  local log_file="${RESULT_ROOT}/logs/${workload}_${safe_policy}_${threads}t.log"
  local trace_file=""

  if [[ "${TRACE_CACHE_ACCESS}" != "0" ]]; then
    trace_file="${result_prefix}.cache_trace.jsonl"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${workload}" \
    "${policy_label}" \
    "${backend}" \
    "${shards}" \
    "${admission}" \
    "${threads}" \
    "${cache_capacity}" \
    "${static_nodes}" \
    "${query_file}" \
    "${truth_file}" \
    "${SEARCH_LIST}" \
    "${BEAM_WIDTH}" \
    "${RECALL_AT}" \
    "${SEARCH_IO_LIMIT}" \
    "${result_prefix}" \
    "${log_file}" \
    "${trace_file}" \
    >> "${MANIFEST}"

  local trace_env=(env)
  if [[ -n "${trace_file}" ]]; then
    trace_env=(env "DISKANN_CACHE_TRACE_PATH=${trace_file}")
  fi

  "${trace_env[@]}" cargo run -p diskann-tools --bin search_disk_index --release -- \
    --data_type "${DATA_TYPE}" \
    --dist_fn "${DIST_FN}" \
    --index_path_prefix "${INDEX_PREFIX}" \
    --result_output_prefix "${result_prefix}" \
    --query_file "${query_file}" \
    --ground_truth_file "${truth_file}" \
    --search_list "${SEARCH_LIST}" \
    --beam_width "${BEAM_WIDTH}" \
    --recall_at "${RECALL_AT}" \
    --search_io_limit "${SEARCH_IO_LIMIT}" \
    --num_threads "${threads}" \
    "$@" \
    2>&1 | tee "${log_file}"
}

for workload in ${WORKLOADS_LIST}; do
  for threads in ${THREADS_LIST}; do
    run_search "${workload}" "NoCache" "none" "0" "none" "${threads}" "0" "0"

    run_search \
      "${workload}" \
      "StaticBFS" \
      "static" \
      "0" \
      "none" \
      "${threads}" \
      "0" \
      "${TOTAL_CACHE_NODES}" \
      --num_nodes_to_cache "${TOTAL_CACHE_NODES}"

    for policy in clock lru; do
      run_search \
        "${workload}" \
        "global_${policy}" \
        "global" \
        "1" \
        "none" \
        "${threads}" \
        "${TOTAL_CACHE_NODES}" \
        "0" \
        --cache_policy "${policy}" \
        --cache_capacity "${TOTAL_CACHE_NODES}"
    done

    for policy in clock fifo lru; do
      run_search \
        "${workload}" \
        "sharded_${policy}" \
        "sharded" \
        "${SHARDS}" \
        "none" \
        "${threads}" \
        "${TOTAL_CACHE_NODES}" \
        "0" \
        --cache_policy "${policy}" \
        --cache_capacity "${TOTAL_CACHE_NODES}" \
        --cache_backend sharded \
        --cache_shards "${SHARDS}"
    done

    run_search \
      "${workload}" \
      "sharded_clock_tiny_lfu" \
      "sharded" \
      "${SHARDS}" \
      "tiny_lfu" \
      "${threads}" \
      "${TOTAL_CACHE_NODES}" \
      "0" \
      --cache_policy clock \
      --cache_capacity "${TOTAL_CACHE_NODES}" \
      --cache_backend sharded \
      --cache_shards "${SHARDS}" \
      --cache_admission tiny_lfu

    for split in ${SPLITS}; do
      static_nodes="${split%%:*}"
      dynamic_nodes="${split##*:}"
      split_label="s${static_nodes}_d${dynamic_nodes}"

      run_search \
        "${workload}" \
        "static_bfs_sharded_clock_${split_label}" \
        "sharded" \
        "${SHARDS}" \
        "none" \
        "${threads}" \
        "${dynamic_nodes}" \
        "${static_nodes}" \
        --cache_policy clock \
        --cache_capacity "${dynamic_nodes}" \
        --cache_backend sharded \
        --cache_shards "${SHARDS}" \
        --cache_static_nodes "${static_nodes}"

      run_search \
        "${workload}" \
        "static_bfs_sharded_clock_tiny_lfu_${split_label}" \
        "sharded" \
        "${SHARDS}" \
        "tiny_lfu" \
        "${threads}" \
        "${dynamic_nodes}" \
        "${static_nodes}" \
        --cache_policy clock \
        --cache_capacity "${dynamic_nodes}" \
        --cache_backend sharded \
        --cache_shards "${SHARDS}" \
        --cache_admission tiny_lfu \
        --cache_static_nodes "${static_nodes}"
    done
  done
done
