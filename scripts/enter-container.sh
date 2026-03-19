#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

"${script_dir}/run-dev-container.sh" >/dev/null
exec "${PODMAN_BIN}" exec -it "${CONTAINER_NAME}" bash -lc "source /etc/profile.d/strix-halo-env.sh && cd /workspace && exec bash"
