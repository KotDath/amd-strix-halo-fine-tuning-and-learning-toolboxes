#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_runtime_dirs

if container_exists; then
  if container_running; then
    echo "${CONTAINER_NAME} is already running"
    exit 0
  fi
  "${PODMAN_BIN}" start "${CONTAINER_NAME}" >/dev/null
  echo "Started existing container ${CONTAINER_NAME}"
  exit 0
fi

"${PODMAN_BIN}" run -d \
  --name "${CONTAINER_NAME}" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add keep-groups \
  --security-opt label=disable \
  --ipc=host \
  --cap-add SYS_PTRACE \
  -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  -e HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" \
  -p "${JUPYTER_BIND_HOST}:${JUPYTER_PORT}:8888" \
  -v "${WORKSPACE_MOUNT}:/workspace:Z" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface:Z" \
  -v "${PIP_CACHE_DIR}:/root/.cache/pip:Z" \
  "${IMAGE_NAME}" >/dev/null

echo "Started ${CONTAINER_NAME}"
