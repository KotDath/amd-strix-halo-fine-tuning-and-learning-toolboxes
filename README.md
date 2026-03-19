# AMD Strix Halo Fine-Tuning And Learning Toolboxes

Reproducible ROCm 7.2 training environment for AMD Strix Halo with:

- PyTorch 2.9.1 on Python 3.12
- `flash-attn` 2.8.4 on the ROCm Triton backend via `aiter`
- `llama.cpp` built from upstream `b8411` for HIP/UMA on `gfx1151`
- `unsloth` pinned to a verified ROCm-compatible commit
- JupyterLab and notebook execution support
- Smoke tests for PyTorch training, `llama.cpp`, and `unsloth` fine-tuning

## Verified Stack

- ROCm userland: 7.2
- Python: 3.12
- PyTorch: `2.9.1+rocm7.2.0`
- Triton: `3.5.1+rocm7.2.0`
- `flash-attn`: `2.8.4`
- `unsloth`: `2026.3.7`
- `unsloth_zoo`: `2026.3.4`
- `transformers`: `5.3.0`
- `peft`: `0.18.1`
- `trl`: `0.23.1`
- `bitsandbytes`: `0.49.2`

## What This Repo Is Good For

- LoRA and QLoRA experimentation on AMD Strix Halo
- supervised fine-tuning in notebooks or CLI scripts
- continued pretraining experiments on smaller models
- `llama.cpp` ROCm inference and device validation
- PyTorch model prototyping on the integrated AMD GPU

## What This Repo Is Not

- a realistic platform for training a competitive large language model from
  scratch on one machine
- a generic multi-vendor container stack
- a minimal image; the base image is intentionally large to prioritize a working
  ROCm path over download size

## Host Requirements

- Fedora host with working `/dev/kfd` and `/dev/dri`
- recommended kernel args for unified memory:
  `iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856`
- `podman` and `toolbox`

Run the host preflight:

```bash
make host-check
```

## Quick Start

```bash
cp .env.example .env
make build
make run
make notebook
```

Then open `http://127.0.0.1:8888/lab`.

The default `.env.example` values already work for the standard local setup.
You only need `.env` if you want to override paths or binaries.

## Ready Scenarios

### I just cloned the repo and want llama.cpp inference

1. Put a `.gguf` model under `models/`.
2. Build the image and start the dev container.
3. Run the inference wrapper.

```bash
make build
make run
make infer MODEL=models/your-model.gguf PROMPT="Hello from Strix Halo"
```

Direct script form:

```bash
./scripts/llama-infer.sh models/your-model.gguf "Hello from Strix Halo"
```

### I want JupyterLab for fine-tuning or training with PyTorch and Unsloth

```bash
make build
make run
make notebook
```

Then open `http://127.0.0.1:8888/lab`.

Recommended first validation:

```bash
make smoke
make smoke-finetune
```

## Main Commands

```bash
make help
make build
make run
make enter
make notebook
make smoke
make smoke-finetune
make smoke-peft
make infer MODEL=models/your-model.gguf PROMPT="Hello"
make collect-env
make toolbox
make stop
make rm
make clean-runtime
```

## Validation Paths

### Main Smoke Suite

```bash
make smoke
```

This validates:

- ROCm-visible PyTorch GPU
- `flash-attn` import and backend selection
- `unsloth` import
- `llama.cpp` ROCm device enumeration
- a small PyTorch training loop on `cuda:0`
- notebook execution through `jupyter nbconvert --execute`

### Unsloth Fine-Tuning Smoke

```bash
make smoke-finetune
```

This runs a real `unsloth` PEFT fine-tuning loop on
`unsloth/gemma-3-270m-it`. The check passes only if:

- the model is on `cuda:0`
- the training batch is on `cuda:0`
- the loss stays finite
- the final evaluation loss is lower than the initial loss

The adapter is saved under `outputs/unsloth_finetune_smoke/adapter`.

### Tiny Causal LM Fine-Tuning Smoke

```bash
make smoke-peft
```

This is a small fallback fine-tuning path for quick regression checks.

## Reports And Artifacts

- `logs/`: notebook and runtime logs
- `reports/`: machine-readable JSON or text reports from smoke tests and
  diagnostics
- `outputs/`: saved adapters and model outputs from smoke or training runs
- `cache/`: Hugging Face and pip caches reused across container runs

Each smoke run writes a JSON report in `reports/` so you can diff regressions
across image rebuilds.

## Project Layout

```text
Dockerfile
versions.env
.env.example
Makefile
scripts/
smoke/
notebooks/
docs/
cache/
logs/
reports/
outputs/
```

## Reproducibility Notes

- ROCm base image, PyTorch wheels, `llama.cpp`, `flash-attn`, and `unsloth`
  commits are pinned in `Dockerfile` and `versions.env`
- shell entrypoints load `versions.env` and optional local overrides from `.env`
- image builds are labeled with the repository revision
- diagnostics can be captured with `make collect-env`

## Notebook And Runtime Notes

- Jupyter runs tokenless on `127.0.0.1` by default and is not exposed to the LAN
- the first `unsloth` fine-tuning run can take a few minutes because of model
  download and first-time ROCm JIT compilation
- the scripts default to `HF_HUB_DISABLE_XET=1` to avoid unstable Xet downloads
  observed during validation

## Toolbox Usage

If you prefer `toolbox` over a long-lived Podman container:

```bash
make toolbox
toolbox enter strix-halo-finetune
```

## Documentation

- `docs/WORKFLOWS.md`
- `docs/TROUBLESHOOTING.md`

## Current Known Gaps

- the repo validates a working `unsloth` fine-tuning path, but it does not yet
  include a polished long-run LoRA notebook for a real dataset
- smoke tests are designed for fast regression detection, not for measuring
  production-quality model quality
