#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

"${script_dir}/run-dev-container.sh" >/dev/null
"${PODMAN_BIN}" exec "${CONTAINER_NAME}" bash -lc "export JUPYTER_PORT=${JUPYTER_PORT} JUPYTER_BIND_HOST=${JUPYTER_BIND_HOST} && cd /workspace && bash /workspace/scripts/container-start-notebook.sh"

echo "JupyterLab is expected at http://${JUPYTER_BIND_HOST}:${JUPYTER_PORT}/lab"
echo "Logs: ${LOG_DIR}/jupyter.log"
