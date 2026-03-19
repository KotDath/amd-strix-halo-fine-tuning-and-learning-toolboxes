#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_runtime_dirs

"${TOOLBOX_BIN}" create strix-halo-finetune \
  --image "${IMAGE_NAME}" \
  -- \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add keep-groups \
  --security-opt label=disable \
  --ipc=host
