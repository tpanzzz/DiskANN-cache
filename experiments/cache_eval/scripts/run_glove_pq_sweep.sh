#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ROOT="${ROOT_DIR}/experiments/cache_eval/results/glove_pq_sweep"
RAW_DATA_ROOT="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/glove_25/data"
RAW_BASE="${RAW_DATA_ROOT}/base.fbin"
RAW_QUERY="${RAW_DATA_ROOT}/query.fbin"
NORMALIZED_BASE="${RUN_ROOT}/data/base.normalized.fbin"
NORMALIZED_QUERY="${RUN_ROOT}/data/query.normalized.fbin"
GROUND_TRUTH="${RAW_DATA_ROOT}/groundtruth.bin"
INDEX_ROOT="${RUN_ROOT}/index"
SEARCH_ROOT="${RUN_ROOT}/search"
SUMMARY="${RUN_ROOT}/glove_pq_sweep_summary.csv"
SEARCH_BINARY="${ROOT_DIR}/target/release/search_disk_index"
L_VALUES=(${L_VALUES:-10 20 30 40 50 75 100 150 200 300 400 600 800 1200 1600})
PQ_CHUNKS_VALUES=(${PQ_CHUNKS_VALUES:-10 16 25 5})
MAX_DEGREE="${MAX_DEGREE:-96}"
BUILD_SEARCH_LIST_SIZE="${BUILD_SEARCH_LIST_SIZE:-200}"
SEARCH_IO_LIMIT="${SEARCH_IO_LIMIT:-2000}"

mkdir -p "${INDEX_ROOT}" "${SEARCH_ROOT}"

for file in "${RAW_BASE}" "${RAW_QUERY}" "${NORMALIZED_BASE}" "${NORMALIZED_QUERY}" "${GROUND_TRUTH}"; do
  if [[ ! -f "${file}" ]]; then
    echo "missing required input: ${file}" >&2
    exit 1
  fi
done

cargo build -q -p diskann-tools --bin build_disk_index --bin generate_pq --bin search_disk_index --release

build_graph() {
  local representation="$1"
  local metric="$2"
  local base_file="$3"
  local prefix="${INDEX_ROOT}/${representation}/glove_${representation}_graph"
  mkdir -p "$(dirname "${prefix}")"
  if [[ ! -f "${prefix}_disk.index" ]]; then
    cargo run -q -p diskann-tools --bin build_disk_index --release -- \
      --data_type float \
      --dist_fn "${metric}" \
      --data_path "${base_file}" \
      --index_path_prefix "${prefix}" \
      --dimensions 25 \
      --full_precision_dimensions 25 \
      --max_degree "${MAX_DEGREE}" \
      --build_search_list_size "${BUILD_SEARCH_LIST_SIZE}" \
      --pq_chunks 5 \
      --index_build_ram_limit_gb 8 >&2 || return 1
  fi
  printf '%s\n' "${prefix}"
}

prepare_pq_variant() {
  local representation="$1"
  local metric="$2"
  local base_file="$3"
  local graph_prefix="$4"
  local chunks="$5"
  local prefix="${INDEX_ROOT}/${representation}/pq${chunks}/glove_${representation}_pq${chunks}"
  mkdir -p "$(dirname "${prefix}")"

  if [[ ! -f "${prefix}_disk.index" ]]; then
    ln "${graph_prefix}_disk.index" "${prefix}_disk.index"
  fi

  if [[ "${chunks}" == "5" ]]; then
    if [[ ! -f "${prefix}_pq_pivots.bin" ]]; then
      ln "${graph_prefix}_pq_pivots.bin" "${prefix}_pq_pivots.bin"
    fi
    if [[ ! -f "${prefix}_pq_compressed.bin" ]]; then
      ln "${graph_prefix}_pq_compressed.bin" "${prefix}_pq_compressed.bin"
    fi
  elif [[ ! -f "${prefix}_pq_pivots.bin" || ! -f "${prefix}_pq_compressed.bin" ]]; then
    cargo run -q -p diskann-tools --bin generate_pq --release -- \
      --data_type float \
      --dist_fn "${metric}" \
      --data_path "${base_file}" \
      --index_path_prefix "${prefix}" \
      --pq_bytes "${chunks}" \
      --num_threads 1 >&2 || return 1
    ln "${prefix}_disk.index_pq_pivots.bin" "${prefix}_pq_pivots.bin"
    ln "${prefix}_disk.index_pq_compressed.bin" "${prefix}_pq_compressed.bin"
  fi
  printf '%s\n' "${prefix}"
}

if [[ ! -f "${SUMMARY}" ]]; then
  echo "representation,metric,navigation,pq_chunks,search_l,qps,mean_latency_us,p999_latency_us,mean_ios,mean_io_us,cpu_us,navigation_preprocess_us,mean_comparisons,mean_hops,recall_at_10_percent,max_rss_kib,navigation_bytes,index_prefix,log_file" > "${SUMMARY}"
fi

run_until_recall() {
  local representation="$1"
  local metric="$2"
  local navigation="$3"
  local chunks="$4"
  local prefix="$5"
  local query_file="$6"
  local full_precision_file="${7:-}"
  local variant="${representation}_${navigation}"
  if [[ "${navigation}" == "pq" ]]; then
    variant="${variant}${chunks}"
  fi
  local output_dir="${SEARCH_ROOT}/${variant}"
  mkdir -p "${output_dir}"

  if awk -F, -v variant="${variant}" 'NR > 1 && ($1 "_" $3 ($3 == "pq" ? $4 : "")) == variant && $15 + 0 >= 95 {found=1} END {exit !found}' "${SUMMARY}"; then
    echo "${variant}: already has a Recall@10 >= 95 result"
    return
  fi

  for l_value in "${L_VALUES[@]}"; do
    if awk -F, -v representation="${representation}" -v navigation="${navigation}" -v chunks="${chunks}" -v l_value="${l_value}" 'NR > 1 && $1 == representation && $3 == navigation && $4 == chunks && $5 == l_value {found=1} END {exit !found}' "${SUMMARY}"; then
      continue
    fi
    local log_file="${output_dir}/l${l_value}.log"
    local result_prefix="${output_dir}/l${l_value}"
    local command=(
      "${SEARCH_BINARY}"
      --data_type float
      --dist_fn "${metric}"
      --index_path_prefix "${prefix}"
      --result_output_prefix "${result_prefix}"
      --query_file "${query_file}"
      --ground_truth_file "${GROUND_TRUTH}"
      --search_list "${l_value}"
      --beam_width 4
      --recall_at 10
      --search_io_limit "${SEARCH_IO_LIMIT}"
      --num_threads 1
      --num_nodes_to_cache 0
    )
    if [[ -n "${full_precision_file}" ]]; then
      command+=(--full_precision_vector_file "${full_precision_file}")
    fi

    /usr/bin/time -v "${command[@]}" 2>&1 | tee "${log_file}"
    local row
    row="$(awk -v l="${l_value}" '$1 == l && $2 == 4 && NF >= 13 {print; exit}' "${log_file}")"
    if [[ -z "${row}" ]]; then
      echo "failed to parse search row from ${log_file}" >&2
      exit 1
    fi
    local qps mean_latency p999 mean_ios mean_io cpu preprocess comparisons hops recall max_rss navigation_bytes
    read -r _ _ qps mean_latency p999 mean_ios mean_io cpu preprocess comparisons hops _ recall <<< "${row}"
    max_rss="$(awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "${log_file}")"
    if [[ "${navigation}" == "full_precision" ]]; then
      navigation_bytes=$((1183514 * 25 * 4))
    else
      navigation_bytes=$((1183514 * chunks))
    fi
    echo "${representation},${metric},${navigation},${chunks},${l_value},${qps},${mean_latency},${p999},${mean_ios},${mean_io},${cpu},${preprocess},${comparisons},${hops},${recall},${max_rss},${navigation_bytes},${prefix},${log_file}" >> "${SUMMARY}"

    if awk -v recall="${recall}" 'BEGIN {exit !(recall + 0 >= 95.0)}'; then
      echo "${variant}: selected L=${l_value}, Recall@10=${recall}"
      return
    fi
  done
  echo "${variant}: no configured L reached Recall@10 >= 95" >&2
}

RAW_GRAPH_PREFIX="$(build_graph raw cosine "${RAW_BASE}")"
NORMALIZED_GRAPH_PREFIX="$(build_graph normalized cosinenormalized "${NORMALIZED_BASE}")"

run_until_recall raw cosine full_precision 0 "${RAW_GRAPH_PREFIX}" "${RAW_QUERY}" "${RAW_BASE}"

for chunks in "${PQ_CHUNKS_VALUES[@]}"; do
  prefix="$(prepare_pq_variant normalized cosinenormalized "${NORMALIZED_BASE}" "${NORMALIZED_GRAPH_PREFIX}" "${chunks}")"
  run_until_recall normalized cosinenormalized pq "${chunks}" "${prefix}" "${NORMALIZED_QUERY}"
done
run_until_recall normalized cosinenormalized full_precision 0 "${NORMALIZED_GRAPH_PREFIX}" "${NORMALIZED_QUERY}" "${NORMALIZED_BASE}"

for chunks in "${PQ_CHUNKS_VALUES[@]}"; do
  prefix="$(prepare_pq_variant raw cosine "${RAW_BASE}" "${RAW_GRAPH_PREFIX}" "${chunks}")"
  run_until_recall raw cosine pq "${chunks}" "${prefix}" "${RAW_QUERY}"
done

echo "summary: ${SUMMARY}"
