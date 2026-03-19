#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

source "${repo_root}/versions.env"

if [[ -f "${repo_root}/.env" ]]; then
  source "${repo_root}/.env"
fi

PODMAN_BIN="${PODMAN_BIN:-podman}"
TOOLBOX_BIN="${TOOLBOX_BIN:-toolbox}"
JUPYTER_BIND_HOST="${JUPYTER_BIND_HOST:-127.0.0.1}"
WORKSPACE_MOUNT="${WORKSPACE_MOUNT:-${repo_root}}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${repo_root}/cache/huggingface}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${repo_root}/cache/pip}"
LOG_DIR="${LOG_DIR:-${repo_root}/logs}"
REPORT_DIR="${REPORT_DIR:-${repo_root}/reports}"
OUTPUT_DIR="${OUTPUT_DIR:-${repo_root}/outputs}"

ensure_runtime_dirs() {
  mkdir -p "${HF_CACHE_DIR}" "${PIP_CACHE_DIR}" "${LOG_DIR}" "${REPORT_DIR}" "${OUTPUT_DIR}"
}

container_exists() {
  "${PODMAN_BIN}" inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

container_running() {
  [[ "$("${PODMAN_BIN}" inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]
}

git_revision() {
  git -C "${repo_root}" rev-parse HEAD 2>/dev/null || echo "unknown"
}
