#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_runtime_dirs
"${script_dir}/run-dev-container.sh" >/dev/null

report_path="${REPORT_DIR}/collect-env-$(date -u +%Y%m%dT%H%M%SZ).txt"

{
  echo "# Strix Halo Environment Report"
  echo
  echo "generated_at_utc=$(date -u --iso-8601=seconds)"
  echo "repo_root=${repo_root}"
  echo "git_revision=$(git_revision)"
  echo "image_name=${IMAGE_NAME}"
  echo "container_name=${CONTAINER_NAME}"
  echo
  echo "## Host"
  "${PODMAN_BIN}" --version
  "${TOOLBOX_BIN}" --version
  echo
  echo "## Image Inspect"
  "${PODMAN_BIN}" image inspect "${IMAGE_NAME}"
  echo
  echo "## Container Inspect"
  "${PODMAN_BIN}" inspect "${CONTAINER_NAME}"
  echo
  echo "## Container Runtime"
  "${PODMAN_BIN}" exec "${CONTAINER_NAME}" bash -lc '
    source /etc/profile.d/strix-halo-env.sh
    echo "python=$(python --version 2>&1)"
    python - <<'"'"'PY'"'"'
import json
import torch

payload = {
    "torch_version": torch.__version__,
    "torch_hip_version": getattr(torch.version, "hip", None),
    "cuda_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(payload, indent=2))
PY
    python -m torch.utils.collect_env
    echo
    echo "llama.cpp devices:"
    llama-cli --list-devices
    echo
    echo "rocminfo:"
    if command -v rocminfo >/dev/null 2>&1; then
      rocminfo
    else
      echo "rocminfo not installed in image"
    fi
  '
} > "${report_path}"

echo "Wrote ${report_path}"
