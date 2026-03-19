#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

if ! container_exists; then
  echo "${CONTAINER_NAME} does not exist"
  exit 0
fi

if container_running; then
  "${PODMAN_BIN}" stop "${CONTAINER_NAME}" >/dev/null
  echo "Stopped ${CONTAINER_NAME}"
  exit 0
fi

echo "${CONTAINER_NAME} is already stopped"
