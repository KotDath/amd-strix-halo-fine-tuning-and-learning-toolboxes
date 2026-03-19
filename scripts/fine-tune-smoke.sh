#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

"${script_dir}/run-dev-container.sh" >/dev/null
"${PODMAN_BIN}" exec "${CONTAINER_NAME}" bash -lc "source /etc/profile.d/strix-halo-env.sh && export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1} HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0} && cd /workspace && python -m smoke.unsloth_finetune_smoke"
