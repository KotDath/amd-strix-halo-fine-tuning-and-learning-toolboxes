#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

"${script_dir}/run-dev-container.sh" >/dev/null
"${PODMAN_BIN}" exec "${CONTAINER_NAME}" bash -lc "source /etc/profile.d/strix-halo-env.sh && cd /workspace && python -m smoke.peft_finetune_smoke"
