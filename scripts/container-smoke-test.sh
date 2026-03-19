#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/strix-halo-env.sh
export LD_LIBRARY_PATH=/usr/local/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}
cd /workspace

python -c 'import torch; print("torch", torch.__version__); print("cuda_available", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0))'
python -c 'import flash_attn; from flash_attn.flash_attn_interface import flash_attn_gpu; print("flash_attn", flash_attn.__version__); print("flash_attn_backend", flash_attn_gpu)'
python -c 'import unsloth; print("unsloth import ok")'
llama-cli --list-devices
python -m smoke.pytorch_smoke_train
jupyter nbconvert --to notebook --execute --inplace notebooks/pytorch_rocm_smoke.ipynb
