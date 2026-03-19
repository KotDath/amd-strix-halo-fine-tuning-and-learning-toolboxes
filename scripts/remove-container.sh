#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

if ! container_exists; then
  echo "${CONTAINER_NAME} does not exist"
  exit 0
fi

if container_running; then
  "${script_dir}/stop-container.sh" >/dev/null
fi

"${PODMAN_BIN}" rm "${CONTAINER_NAME}" >/dev/null
echo "Removed ${CONTAINER_NAME}"
