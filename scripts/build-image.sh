#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_runtime_dirs
revision="$(git_revision)"

"${PODMAN_BIN}" build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg LLAMA_CPP_TAG="${LLAMA_CPP_TAG}" \
  --build-arg LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT}" \
  --build-arg FLASH_ATTENTION_TAG="${FLASH_ATTENTION_TAG}" \
  --build-arg FLASH_ATTENTION_AITER_COMMIT="${FLASH_ATTENTION_AITER_COMMIT}" \
  --build-arg FLASH_ATTENTION_AITER_CK_COMMIT="${FLASH_ATTENTION_AITER_CK_COMMIT}" \
  --build-arg UNSLOTH_COMMIT="${UNSLOTH_COMMIT}" \
  --build-arg UNSLOTH_ZOO_COMMIT="${UNSLOTH_ZOO_COMMIT}" \
  --label org.opencontainers.image.title="AMD Strix Halo Fine-Tuning And Learning Toolboxes" \
  --label org.opencontainers.image.revision="${revision}" \
  --label org.opencontainers.image.source="${repo_root}" \
  -t "${IMAGE_NAME}" \
  -f "${repo_root}/Dockerfile" \
  "${repo_root}"
