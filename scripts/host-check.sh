#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_runtime_dirs

required_args=(
  "iommu=pt"
  "amdgpu.gttsize=126976"
  "ttm.pages_limit=32505856"
)

echo "Checking host prerequisites..."
command -v "${PODMAN_BIN}" >/dev/null
command -v "${TOOLBOX_BIN}" >/dev/null
[[ -e /dev/kfd ]]
[[ -d /dev/dri ]]

cmdline="$(< /proc/cmdline)"
missing_args=()
for arg in "${required_args[@]}"; do
  if [[ "${cmdline}" != *"${arg}"* ]]; then
    missing_args+=("${arg}")
  fi
done

echo "podman: $("${PODMAN_BIN}" --version)"
echo "toolbox: $("${TOOLBOX_BIN}" --version)"
echo "devices:"
ls -l /dev/kfd /dev/dri
echo "kernel cmdline: ${cmdline}"
echo "disk:"
df -h "${repo_root}"
echo "memory:"
free -h

if (( ${#missing_args[@]} > 0 )); then
  printf 'Missing recommended kernel args: %s\n' "${missing_args[*]}" >&2
  exit 1
fi

echo "Host checks passed"
