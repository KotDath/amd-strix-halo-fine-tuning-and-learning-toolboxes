#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

"${script_dir}/run-dev-container.sh" >/dev/null
"${PODMAN_BIN}" exec "${CONTAINER_NAME}" bash -lc "cd /workspace && bash /workspace/scripts/container-smoke-test.sh"
