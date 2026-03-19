#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

if [[ $# -lt 1 ]]; then
  cat <<'EOF' >&2
Usage:
  ./scripts/llama-infer.sh <model-path> [prompt]

Examples:
  ./scripts/llama-infer.sh models/qwen2.5-0.5b-instruct-q4_k_m.gguf "Привет"
  PROMPT="Summarize ROCm UMA in one sentence." ./scripts/llama-infer.sh models/model.gguf

Notes:
  - The model file must live inside this repository, for example under models/.
  - Extra llama.cpp flags can be passed via LLAMA_ARGS.
EOF
  exit 1
fi

model_arg="$1"
shift || true
prompt_arg="${1:-${PROMPT:-Hello from Strix Halo.}}"

if [[ "${model_arg}" = /* ]]; then
  case "${model_arg}" in
    "${repo_root}"/*) ;;
    *)
      echo "Absolute model path must be inside ${repo_root}" >&2
      exit 1
      ;;
  esac
  model_host_path="${model_arg}"
  model_container_path="/workspace/${model_arg#${repo_root}/}"
else
  model_host_path="${repo_root}/${model_arg}"
  model_container_path="/workspace/${model_arg}"
fi

if [[ ! -f "${model_host_path}" ]]; then
  echo "Model file not found: ${model_host_path}" >&2
  exit 1
fi

llama_args="${LLAMA_ARGS:-}"

"${script_dir}/run-dev-container.sh" >/dev/null
"${PODMAN_BIN}" exec "${CONTAINER_NAME}" bash -lc "
  source /etc/profile.d/strix-halo-env.sh
  cd /workspace
  llama-cli \
    -m \"${model_container_path}\" \
    -p \"${prompt_arg}\" \
    -ngl 999 \
    -c 4096 \
    -n 128 \
    ${llama_args}
"
