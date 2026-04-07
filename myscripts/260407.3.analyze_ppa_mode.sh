#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for mode-specific PPA/proxy analysis.
# Usage:
#   bash myscripts/260407.3.analyze_ppa_mode.sh GP
#   bash myscripts/260407.3.analyze_ppa_mode.sh MP
# Optional env overrides:
#   SEEDS=1,2,3 FORMULATIONS=MGO,HPO GLOBAL_CORR_MODE=design_mean

MODE="${1:-GP}"
SEEDS="${SEEDS:-1,2,3}"
FORMULATIONS="${FORMULATIONS:-MGO,HPO}"
BENCHMARKS="${BENCHMARKS:-OpenROAD}"
GLOBAL_CORR_MODE="${GLOBAL_CORR_MODE:-design_mean}"
ONLY_EVAL_OK="${ONLY_EVAL_OK:-true}"

SEED_TAG="${SEEDS//,/_}"
MODE_TAG="$(echo "${MODE}" | tr '[:upper:]' '[:lower:]' | tr ',' '_')"

PPA_CSV="${PPA_CSV:-results/analysis_reports/ppa_eval/seeds_${SEED_TAG}/ppa_eval_seeds_${SEED_TAG}_${MODE_TAG}_best.csv}"
HV_JSON="${HV_JSON:-results/analysis_reports/hv/json/hv_summary_seeds_${SEED_TAG}.json}"
OUT_DIR="${OUT_DIR:-results/analysis_reports/ppa_proxy/study_seeds_${SEED_TAG}_${MODE_TAG}}"

CMD=(
  python3 src/utils/analyze_ppa_proxy.py
  --ppa_csv "${PPA_CSV}"
  --hv_json "${HV_JSON}"
  --output_dir "${OUT_DIR}"
  --benchmarks "${BENCHMARKS}"
  --formulations "${FORMULATIONS}"
  --modes "${MODE}"
  --global_corr_mode "${GLOBAL_CORR_MODE}"
)

if [[ "${ONLY_EVAL_OK}" == "true" ]]; then
  CMD+=(--only_eval_ok)
fi

echo "[RUN] ${CMD[*]}"
"${CMD[@]}"
