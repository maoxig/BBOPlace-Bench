#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for mode-specific PPA evaluation.
# Usage:
#   bash myscripts/260407.2.eval_ppa_mode.sh GP
#   bash myscripts/260407.2.eval_ppa_mode.sh MP
# Optional env overrides:
#   SEEDS=1,2,3 JOBS=4 NUM_SHARDS=3 SHARD_ID=1 WORKER_TAG=m1 FORMULATIONS=MGO,HPO

MODE="${1:-GP}"
SEEDS="${SEEDS:-1,2,3}"
JOBS="${JOBS:-1}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-1}"
WORKER_TAG="${WORKER_TAG:-}"
FORMULATIONS="${FORMULATIONS:-MGO,HPO}"
BENCHMARKS="${BENCHMARKS:-OpenROAD}"
OUTPUT_BASE="${OUTPUT_BASE:-results/analysis_reports/ppa_eval}"
HV_JSON="${HV_JSON:-results/analysis_reports/hv/json/hv_summary_seeds_${SEEDS//,/_}.json}"

CMD=(
  python3 src/utils/evaluate_best_gp_ppa.py
  --workspace .
  --output "${OUTPUT_BASE}"
  --seeds "${SEEDS}"
  --hv_json "${HV_JSON}"
  --benchmarks "${BENCHMARKS}"
  --formulations "${FORMULATIONS}"
  --modes "${MODE}"
  --jobs "${JOBS}"
  --num_shards "${NUM_SHARDS}"
  --shard_id "${SHARD_ID}"
  --reuse_existing_results true
)

if [[ -n "${WORKER_TAG}" ]]; then
  CMD+=(--worker_tag "${WORKER_TAG}")
fi

echo "[RUN] ${CMD[*]}"
"${CMD[@]}"
