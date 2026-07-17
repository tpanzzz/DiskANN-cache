#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CATALOG="${ROOT_DIR}/experiments/cache_eval/datasets/catalog.json"
PYTHON="${ROOT_DIR}/experiments/cache_eval/.venv/bin/python"
DATASET="${1:-}"

if [[ -z "${DATASET}" ]]; then
  echo "usage: $0 <catalog-dataset-name>" >&2
  exit 2
fi

readarray -t SPEC < <(
  "${PYTHON}" - "${CATALOG}" "${DATASET}" "${ROOT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

catalog_path = Path(sys.argv[1])
name = sys.argv[2]
root = Path(sys.argv[3])
catalog = json.loads(catalog_path.read_text())
if name not in catalog:
    raise SystemExit(f"unknown dataset {name!r}")
spec = catalog[name]
print((root / spec["local_hdf5"]).resolve())
print(spec["dimensions"])
print(spec["pq_chunks"])
print(spec["diskann_metric"])
print(spec["base_count"])
PY
)

HDF5_PATH="${SPEC[0]}"
DIMENSIONS="${SPEC[1]}"
PQ_CHUNKS="${PQ_CHUNKS_OVERRIDE:-${SPEC[2]}}"
METRIC="${SPEC[3]}"
BASE_COUNT="${SPEC[4]}"

RUN_ROOT="${ROOT_DIR}/experiments/cache_eval/results/spatial_locality_pilot/${DATASET}"
DATA_DIR="${RUN_ROOT}/data"
MAX_DEGREE="${MAX_DEGREE:-64}"
BUILD_SEARCH_LIST_SIZE="${BUILD_SEARCH_LIST_SIZE:-100}"
INDEX_TAG="${INDEX_TAG:-r${MAX_DEGREE}_l${BUILD_SEARCH_LIST_SIZE}}"
INDEX_PREFIX="${RUN_ROOT}/index/${DATASET}_${INDEX_TAG}"
SEARCH_DIR="${RUN_ROOT}/search/${INDEX_TAG}"
TRACE_DIR="${RUN_ROOT}/expansions/${INDEX_TAG}"
mkdir -p "${DATA_DIR}" "$(dirname "${INDEX_PREFIX}")" "${SEARCH_DIR}" "${TRACE_DIR}"

BASE_FILE="${DATA_DIR}/base.fbin"
QUERY_FILE="${DATA_DIR}/query.fbin"
GROUNDTRUTH_FILE="${DATA_DIR}/groundtruth.bin"

if [[ ! -f "${BASE_FILE}" || ! -f "${QUERY_FILE}" || ! -f "${GROUNDTRUTH_FILE}" ]]; then
  "${PYTHON}" "${ROOT_DIR}/experiments/cache_eval/hdf5_to_diskann_bin.py" \
    --input "${HDF5_PATH}" \
    --output-dir "${DATA_DIR}"
fi

if [[ ! -f "${INDEX_PREFIX}_disk.index" ]]; then
  cargo run -q -p diskann-tools --bin build_disk_index --release -- \
    --data_type float \
    --dist_fn "${METRIC}" \
    --data_path "${BASE_FILE}" \
    --index_path_prefix "${INDEX_PREFIX}" \
    --dimensions "${DIMENSIONS}" \
    --full_precision_dimensions "${DIMENSIONS}" \
    --max_degree "${MAX_DEGREE}" \
    --build_search_list_size "${BUILD_SEARCH_LIST_SIZE}" \
    --pq_chunks "${PQ_CHUNKS}" \
    --index_build_ram_limit_gb 8
fi

SEARCH_LISTS="${SEARCH_LISTS:-50,100,200,400}"
SEARCH_IO_LIMIT="${SEARCH_IO_LIMIT:-1000}"
SEARCH_LOG="${SEARCH_DIR}/recall_sweep.log"
cargo run -q -p diskann-tools --bin search_disk_index --release -- \
  --data_type float \
  --dist_fn "${METRIC}" \
  --index_path_prefix "${INDEX_PREFIX}" \
  --result_output_prefix "${SEARCH_DIR}/recall_sweep" \
  --query_file "${QUERY_FILE}" \
  --ground_truth_file "${GROUNDTRUTH_FILE}" \
  --search_list "${SEARCH_LISTS}" \
  --beam_width 4 \
  --recall_at 10 \
  --search_io_limit "${SEARCH_IO_LIMIT}" \
  --num_threads 1 \
  --num_nodes_to_cache 0 \
  2>&1 | tee "${SEARCH_LOG}"

SELECTED_L="$(awk '$1 ~ /^[0-9]+$/ && $2 == 4 && NF >= 13 && ($13 + 0) >= 95.0 {print $1; exit}' "${SEARCH_LOG}")"
if [[ -z "${SELECTED_L}" ]]; then
  echo "no search-list value reached Recall@10 >= 95; see ${SEARCH_LOG}" >&2
  exit 1
fi
echo "selected search_list=${SELECTED_L} for expansion trace"

TRACE_FILE="${TRACE_DIR}/cache_access_trace.jsonl"
TRACE_LOG="${TRACE_DIR}/trace_search.log"
DISKANN_CACHE_TRACE_PATH="${TRACE_FILE}" \
cargo run -q -p diskann-tools --bin search_disk_index --release -- \
  --data_type float \
  --dist_fn "${METRIC}" \
  --index_path_prefix "${INDEX_PREFIX}" \
  --result_output_prefix "${TRACE_DIR}/trace_search" \
  --query_file "${QUERY_FILE}" \
  --ground_truth_file "${GROUNDTRUTH_FILE}" \
  --search_list "${SELECTED_L}" \
  --beam_width 4 \
  --recall_at 10 \
  --search_io_limit "${SEARCH_IO_LIMIT}" \
  --num_threads 1 \
  --cache_policy lru \
  --cache_capacity 1 \
  2>&1 | tee "${TRACE_LOG}"

"${PYTHON}" "${ROOT_DIR}/experiments/cache_eval/scripts/summarize_node_expansions.py" \
  --trace "${TRACE_FILE}" \
  --num-nodes "${BASE_COUNT}" \
  --output-dir "${TRACE_DIR}/summary"

echo "pilot outputs: ${RUN_ROOT}"
