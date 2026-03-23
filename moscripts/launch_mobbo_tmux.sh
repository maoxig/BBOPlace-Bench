#!/usr/bin/env bash
set -euo pipefail

# Batch launcher for scripts under moscripts/ into one tmux session.
#
# Defaults are tuned for your request:
# - session: MOBBO
# - one window per script
# - windows named by algo+placer+mode (and benchmark suffix when needed)
#
# Example:
#   bash moscripts/launch_mobbo_tmux.sh
#   SESSION_NAME=MOBBO ROUND_LABEL=seed2 bash moscripts/launch_mobbo_tmux.sh
#   FILTER_BENCHMARKS=ICCAD2015 FILTER_MODES=GP bash moscripts/launch_mobbo_tmux.sh
#   DRY_RUN=1 bash moscripts/launch_mobbo_tmux.sh

SESSION_NAME="${SESSION_NAME:-MOBBO}"
SCRIPTS_DIR="${SCRIPTS_DIR:-moscripts}"
SCRIPT_GLOB="${SCRIPT_GLOB:-MO_*.sh}"
FILTER_BENCHMARKS="${FILTER_BENCHMARKS:-all}"   # all or comma list, e.g. ICCAD2015,OPENROAD
FILTER_MODES="${FILTER_MODES:-all}"             # all or comma list, e.g. GP,MP
FILTER_PLACERS="${FILTER_PLACERS:-all}"         # all or comma list, e.g. HPO,MGO
FILTER_ALGOS="${FILTER_ALGOS:-all}"             # all or comma list, e.g. NSGA2,MOEAD
ROUND_LABEL="${ROUND_LABEL:-seed2}"             # shown in command output only
WINDOW_NAME_STYLE="${WINDOW_NAME_STYLE:-apm}"   # apm or benchmark_apm
AUTO_APPEND_BENCH_ON_CONFLICT="${AUTO_APPEND_BENCH_ON_CONFLICT:-1}"
START_INDEX="${START_INDEX:-0}"                 # skip first N selected scripts
LIMIT="${LIMIT:-0}"                             # 0 means no limit
DRY_RUN="${DRY_RUN:-0}"
ATTACH="${ATTACH:-0}"
ENABLE_DOCKER="${ENABLE_DOCKER:-0}"            # 1: run each task through docker wrapper
DOCKER_WRAPPER_SCRIPT="${DOCKER_WRAPPER_SCRIPT:-moscripts/docker_run_wrapper.sh}"
TMUX_REMAIN_ON_EXIT="${TMUX_REMAIN_ON_EXIT:-1}" # keep dead windows for debugging
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"            # optional, e.g. base

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/${SCRIPTS_DIR}"
DOCKER_WRAPPER_PATH="${ROOT_DIR}/${DOCKER_WRAPPER_SCRIPT}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux not found in PATH" >&2
  exit 1
fi

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "[ERROR] scripts directory not found: ${TARGET_DIR}" >&2
  exit 1
fi

if [[ "${ENABLE_DOCKER}" == "1" && ! -f "${DOCKER_WRAPPER_PATH}" ]]; then
  echo "[ERROR] docker wrapper not found: ${DOCKER_WRAPPER_PATH}" >&2
  exit 1
fi

trim() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf '%s' "${s}"
}

to_lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

in_csv_or_all() {
  local value="$(to_lower "$1")"
  local csv="$(to_lower "$(trim "$2")")"
  if [[ "${csv}" == "all" || -z "${csv}" ]]; then
    return 0
  fi

  local item
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    item="$(trim "${item}")"
    if [[ -n "${item}" && "${item}" == "${value}" ]]; then
      return 0
    fi
  done
  return 1
}

parse_script_name() {
  # Expected: MO_<BENCHMARK>_<PLACER>_<ALGO>_<MODE>.sh
  local file_name="$1"
  local stem="${file_name%.sh}"
  IFS='_' read -r prefix benchmark placer algo mode <<< "${stem}"

  if [[ "${prefix}" != "MO" || -z "${benchmark:-}" || -z "${placer:-}" || -z "${algo:-}" || -z "${mode:-}" ]]; then
    return 1
  fi

  printf '%s;%s;%s;%s\n' "${benchmark}" "${placer}" "${algo}" "${mode}"
}

compose_window_name() {
  local benchmark="$1"
  local placer="$2"
  local algo="$3"
  local mode="$4"

  local apm="$(to_lower "${algo}")_$(to_lower "${placer}")_$(to_lower "${mode}")"
  if [[ "${WINDOW_NAME_STYLE}" == "benchmark_apm" ]]; then
    printf '%s' "$(to_lower "${benchmark}")_${apm}"
  else
    printf '%s' "${apm}"
  fi
}

mapfile -t all_scripts < <(cd "${TARGET_DIR}" && ls ${SCRIPT_GLOB} 2>/dev/null | sort)
if [[ ${#all_scripts[@]} -eq 0 ]]; then
  echo "[ERROR] no scripts matched: ${TARGET_DIR}/${SCRIPT_GLOB}" >&2
  exit 1
fi

declare -a selected=()
for script in "${all_scripts[@]}"; do
  meta="$(parse_script_name "${script}" || true)"
  if [[ -z "${meta}" ]]; then
    continue
  fi

  IFS=';' read -r benchmark placer algo mode <<< "${meta}"
  in_csv_or_all "${benchmark}" "${FILTER_BENCHMARKS}" || continue
  in_csv_or_all "${mode}" "${FILTER_MODES}" || continue
  in_csv_or_all "${placer}" "${FILTER_PLACERS}" || continue
  in_csv_or_all "${algo}" "${FILTER_ALGOS}" || continue

  selected+=("${script}")
done

if [[ ${#selected[@]} -eq 0 ]]; then
  echo "[ERROR] no scripts selected after filters" >&2
  exit 1
fi

if [[ "${START_INDEX}" -gt 0 ]]; then
  if [[ "${START_INDEX}" -ge ${#selected[@]} ]]; then
    echo "[ERROR] START_INDEX (${START_INDEX}) >= selected scripts (${#selected[@]})" >&2
    exit 1
  fi
  selected=("${selected[@]:${START_INDEX}}")
fi

if [[ "${LIMIT}" -gt 0 && "${LIMIT}" -lt ${#selected[@]} ]]; then
  selected=("${selected[@]:0:${LIMIT}}")
fi

echo "[INFO] session       : ${SESSION_NAME}"
echo "[INFO] scripts dir   : ${TARGET_DIR}"
echo "[INFO] round label   : ${ROUND_LABEL}"
echo "[INFO] selected count: ${#selected[@]}"
echo "[INFO] use docker    : ${ENABLE_DOCKER}"
echo "[INFO] remain-on-exit: ${TMUX_REMAIN_ON_EXIT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] DRY_RUN=1, no tmux command will be executed"
fi

if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] tmux new-session -d -s ${SESSION_NAME}"
  else
    tmux new-session -d -s "${SESSION_NAME}"
    tmux rename-window -t "${SESSION_NAME}:0" "launcher"
  fi
else
  echo "[WARN] tmux session already exists: ${SESSION_NAME}"
fi

if [[ "${TMUX_REMAIN_ON_EXIT}" == "1" ]]; then
  tmux set-option -t "${SESSION_NAME}" remain-on-exit on
fi

declare -A win_used=()

for script in "${selected[@]}"; do
  meta="$(parse_script_name "${script}")"
  IFS=';' read -r benchmark placer algo mode <<< "${meta}"

  window_name="$(compose_window_name "${benchmark}" "${placer}" "${algo}" "${mode}")"
  if [[ -n "${win_used[${window_name}]:-}" && "${AUTO_APPEND_BENCH_ON_CONFLICT}" == "1" ]]; then
    window_name="${window_name}_$(to_lower "${benchmark}")"
  fi
  if [[ -n "${win_used[${window_name}]:-}" ]]; then
    suffix=2
    while [[ -n "${win_used[${window_name}_${suffix}]:-}" ]]; do
      ((suffix++))
    done
    window_name="${window_name}_${suffix}"
  fi
  win_used["${window_name}"]=1

  # Run from moscripts so relative paths in target scripts keep working.
  if [[ "${ENABLE_DOCKER}" == "1" ]]; then
    full_cmd="bash '${DOCKER_WRAPPER_PATH}' '${script}' '${TARGET_DIR}' '${ROOT_DIR}' '${ROUND_LABEL}' '${window_name}'"
  else
    full_cmd="cd '${TARGET_DIR}' && echo '[START][${ROUND_LABEL}] ${script}' && bash '${script}' 2>&1 | tee -a '${script%.sh}.${ROUND_LABEL}.log'"
  fi

  if [[ -n "${CONDA_ENV_NAME}" ]]; then
    full_cmd="source \"\$HOME/.bashrc\" >/dev/null 2>&1 || true && conda activate '${CONDA_ENV_NAME}' && ${full_cmd}"
  fi

  quoted_cmd="$(printf '%q' "${full_cmd}")"
  cmd="bash -lc ${quoted_cmd}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] window=${window_name} script=${script}"
    echo "          cmd=${cmd}"
    continue
  fi

  if tmux new-window -t "${SESSION_NAME}" -n "${window_name}" "${cmd}"; then
    echo "[OK] started ${script} -> ${SESSION_NAME}:${window_name}"
  else
    echo "[ERROR] failed to start ${script} in window ${window_name}" >&2
  fi
done

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "[INFO] all selected scripts started in session: ${SESSION_NAME}"
  echo "[INFO] view windows: tmux list-windows -t ${SESSION_NAME}"
  tmux list-windows -t "${SESSION_NAME}" || true
  if [[ "${ATTACH}" == "1" ]]; then
    tmux attach -t "${SESSION_NAME}"
  else
    echo "[INFO] attach manually: tmux attach -t ${SESSION_NAME}"
  fi
fi
