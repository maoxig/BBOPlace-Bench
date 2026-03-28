#!/usr/bin/env bash
set -euo pipefail

# Server-customizable docker wrapper.
# Called by launch_mobbo_tmux.sh when ENABLE_DOCKER=1.
# Behavior:
#   - Reuse one shared container instance.
#   - First call creates/starts container.
#   - Following calls execute commands via docker exec.
#
# Args:
#   1: script name, e.g. MO_ICCAD2015_MGO_NSGA2_GP.sh
#   2: target dir (moscripts absolute path)
#   3: workspace root absolute path
#   4: round label
#   5: window name
#   6: log file absolute path
#   7: tmp dir
#   8: wandb dir
#   9: wandb cache dir
#  10: wandb config dir
#  11: wandb data dir

SCRIPT_NAME="${1:?missing script name}"
TARGET_DIR="${2:?missing target dir}"
ROOT_DIR="${3:?missing root dir}"
ROUND_LABEL="${4:?missing round label}"
WINDOW_NAME="${5:?missing window name}"
LOG_FILE="${6:?missing log file}"
TMP_DIR="${7:-}"
WANDB_DIR_PATH="${8:-}"
WANDB_CACHE_DIR_PATH="${9:-}"
WANDB_CONFIG_DIR_PATH="${10:-}"
WANDB_DATA_DIR_PATH="${11:-}"

DOCKER_IMAGE="${DOCKER_IMAGE:-crt/bboplace-bench:cuda}"
DOCKER_CONTAINER_PREFIX="${DOCKER_CONTAINER_PREFIX:-mo-bbo}"
DOCKER_CONTAINER_NAME="${DOCKER_CONTAINER_NAME:-${DOCKER_CONTAINER_PREFIX}}"
DOCKER_NETWORK_MODE="${DOCKER_NETWORK_MODE:-host}"
DOCKER_USE_PRIVILEGED="${DOCKER_USE_PRIVILEGED:-1}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
DOCKER_KEEP_ALIVE_CMD="${DOCKER_KEEP_ALIVE_CMD:-tail -f /dev/null}"
DOCKER_WORKSPACE_DIR="${DOCKER_WORKSPACE_DIR:-/workspace}"

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -Fxq "${DOCKER_CONTAINER_NAME}"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -Fxq "${DOCKER_CONTAINER_NAME}"
}

host_to_container_path() {
  local host_path="$1"
  if [[ -z "${host_path}" ]]; then
    printf ''
    return
  fi

  if [[ "${host_path}" == "${ROOT_DIR}" ]]; then
    printf '%s' "${DOCKER_WORKSPACE_DIR}"
    return
  fi

  if [[ "${host_path}" == "${ROOT_DIR}/"* ]]; then
    printf '%s' "${DOCKER_WORKSPACE_DIR}${host_path#${ROOT_DIR}}"
    return
  fi

  printf '%s' "${host_path}"
}

container_workspace_mounted() {
  docker inspect "${DOCKER_CONTAINER_NAME}" --format '{{range .Mounts}}{{if and (eq .Source "'"${ROOT_DIR}"'") (eq .Destination "'"${DOCKER_WORKSPACE_DIR}"'")}}ok{{end}}{{end}}' 2>/dev/null | grep -q '^ok$'
}

CONTAINER_TARGET_DIR="$(host_to_container_path "${TARGET_DIR}")"
CONTAINER_LOG_FILE="$(host_to_container_path "${LOG_FILE}")"
CONTAINER_TMP_DIR="$(host_to_container_path "${TMP_DIR}")"
CONTAINER_WANDB_DIR="$(host_to_container_path "${WANDB_DIR_PATH}")"
CONTAINER_WANDB_CACHE_DIR="$(host_to_container_path "${WANDB_CACHE_DIR_PATH}")"
CONTAINER_WANDB_CONFIG_DIR="$(host_to_container_path "${WANDB_CONFIG_DIR_PATH}")"
CONTAINER_WANDB_DATA_DIR="$(host_to_container_path "${WANDB_DATA_DIR_PATH}")"

docker_args=(
  run -d
  --name "${DOCKER_CONTAINER_NAME}"
  --gpus "${DOCKER_GPUS}"
  -u "$(id -u):$(id -g)"
  -v /etc/passwd:/etc/passwd:ro
  -v /etc/group:/etc/group:ro
  -v "${ROOT_DIR}:${DOCKER_WORKSPACE_DIR}"
  -w "${DOCKER_WORKSPACE_DIR}"
  -v "${HOME}:${HOME}"
  -v "${HOME}/.bashrc:${HOME}/.bashrc"
  -v "${HOME}/.profile:${HOME}/.profile"
  --network "${DOCKER_NETWORK_MODE}"
)

if [[ "${DOCKER_USE_PRIVILEGED}" == "1" ]]; then
  docker_args+=(--privileged)
fi

inner_cmd="set -m; trap 'jobs -pr | xargs -r kill -TERM >/dev/null 2>&1 || true' INT TERM; mkdir -p '${CONTAINER_TMP_DIR}' '${CONTAINER_WANDB_DIR}' '${CONTAINER_WANDB_CACHE_DIR}' '${CONTAINER_WANDB_CONFIG_DIR}' '${CONTAINER_WANDB_DATA_DIR}' && export TMPDIR='${CONTAINER_TMP_DIR}' TEMP='${CONTAINER_TMP_DIR}' TMP='${CONTAINER_TMP_DIR}' WANDB_DIR='${CONTAINER_WANDB_DIR}' WANDB_CACHE_DIR='${CONTAINER_WANDB_CACHE_DIR}' WANDB_CONFIG_DIR='${CONTAINER_WANDB_CONFIG_DIR}' WANDB_DATA_DIR='${CONTAINER_WANDB_DATA_DIR}' && cd '${CONTAINER_TARGET_DIR}' && echo '[START][${ROUND_LABEL}] ${SCRIPT_NAME} (docker)' && bash '${SCRIPT_NAME}' 2>&1 | tee -a '${CONTAINER_LOG_FILE}'"

if container_running; then
  if ! container_workspace_mounted; then
    echo "[ERROR] container ${DOCKER_CONTAINER_NAME} exists but ${ROOT_DIR} is not mounted to ${DOCKER_WORKSPACE_DIR}" >&2
    echo "        remove and recreate it: docker rm -f ${DOCKER_CONTAINER_NAME}" >&2
    exit 1
  fi
elif container_exists; then
  docker start "${DOCKER_CONTAINER_NAME}" >/dev/null
  if ! container_workspace_mounted; then
    echo "[ERROR] container ${DOCKER_CONTAINER_NAME} exists but ${ROOT_DIR} is not mounted to ${DOCKER_WORKSPACE_DIR}" >&2
    echo "        remove and recreate it: docker rm -f ${DOCKER_CONTAINER_NAME}" >&2
    exit 1
  fi
else
  docker "${docker_args[@]}" "${DOCKER_IMAGE}" /bin/bash -lc "${DOCKER_KEEP_ALIVE_CMD}" >/dev/null
fi

exec_io=(-i)
if [[ -t 0 && -t 1 ]]; then
  exec_io=(-it)
fi

docker exec "${exec_io[@]}" "${DOCKER_CONTAINER_NAME}" /bin/bash -lc "${inner_cmd}"
