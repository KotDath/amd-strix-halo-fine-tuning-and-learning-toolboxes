#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

rm -rf "${LOG_DIR}" "${REPORT_DIR}" "${repo_root}/unsloth_compiled_cache"
rm -rf \
  "${OUTPUT_DIR}/peft_finetune_smoke" \
  "${OUTPUT_DIR}/unsloth_finetune_smoke"
mkdir -p "${LOG_DIR}" "${REPORT_DIR}"
echo "Cleaned runtime logs, reports, smoke outputs, and unsloth compiled cache under ${repo_root}"
