# Workflows

## Ready Scenarios

### I just cloned the repo and want to run inference via llama.cpp

1. Build the image.
2. Put a GGUF model inside `models/`.
3. Run inference through the wrapper script or `make infer`.

```bash
make build
make run
make infer MODEL=models/your-model.gguf PROMPT="Hello from Strix Halo"
```

You can also call the script directly:

```bash
./scripts/llama-infer.sh models/your-model.gguf "Hello from Strix Halo"
```

### I want to start JupyterLab for fine-tuning or training with PyTorch and Unsloth

1. Build the image once.
2. Start the long-lived dev container.
3. Start JupyterLab.
4. Open the notebook UI locally.

```bash
make build
make run
make notebook
```

Then open `http://127.0.0.1:8888/lab`.

Recommended first validation after bring-up:

```bash
make smoke
make smoke-finetune
```

## 5-Minute Bring-Up

```bash
make host-check
make build
make run
make notebook
make smoke
make smoke-finetune
```

## Main Modes

### Notebook-first experimentation

Use this when you want interactive training, data inspection, and ad hoc model
debugging:

```bash
make run
make notebook
```

Then open `http://127.0.0.1:8888/lab`.

### CLI smoke validation

Use this after every image rebuild:

```bash
make smoke
make smoke-finetune
```

### Diagnostics snapshot

Use this when you need a reproducible environment report for bug reports or
regression tracking:

```bash
make collect-env
```

## Recommended Training Progression

1. Validate PyTorch and ROCm with `make smoke`.
2. Validate a small `unsloth` PEFT path with `make smoke-finetune`.
3. Prototype dataset formatting and batching in a notebook.
4. Run a longer LoRA or continued-pretraining job with checkpointing enabled.
5. Export artifacts under `outputs/` and validate them with `llama.cpp` or a
   Transformers inference script.

## Scope Guidance

- Use this repo for LoRA, QLoRA, SFT, continued pretraining, and small-to-medium
  custom transformer experiments.
- Do not treat a single Strix Halo system as a realistic platform for training a
  competitive modern multi-billion-parameter LLM from scratch.
