SHELL := /usr/bin/env bash

.PHONY: help host-check build run enter notebook smoke smoke-finetune smoke-peft collect-env toolbox stop rm clean-runtime infer

help:
	@printf '%s\n' \
	  'Available targets:' \
	  '  make host-check       Verify host prerequisites' \
	  '  make build            Build the container image' \
	  '  make run              Start the long-lived dev container' \
	  '  make enter            Open a shell inside the dev container' \
	  '  make notebook         Start JupyterLab inside the container' \
	  '  make smoke            Run the main ROCm smoke test suite' \
	  '  make smoke-finetune   Run the Unsloth fine-tuning smoke test' \
	  '  make smoke-peft       Run the tiny causal LM fine-tuning smoke test' \
	  '  make infer MODEL=... [PROMPT=...]  Run llama.cpp inference on a GGUF model in models/' \
	  '  make collect-env      Write a diagnostic environment report' \
	  '  make toolbox          Create a toolbox from the built image' \
	  '  make stop             Stop the dev container' \
	  '  make rm               Remove the dev container' \
	  '  make clean-runtime    Remove runtime logs, reports, and JIT cache'

host-check:
	./scripts/host-check.sh

build:
	./scripts/build-image.sh

run:
	./scripts/run-dev-container.sh

enter:
	./scripts/enter-container.sh

notebook:
	./scripts/start-notebook.sh

smoke:
	./scripts/smoke-test.sh

smoke-finetune:
	./scripts/fine-tune-smoke.sh

smoke-peft:
	./scripts/peft-fine-tune-smoke.sh

infer:
	@test -n "$(MODEL)" || (echo "Usage: make infer MODEL=models/your-model.gguf [PROMPT='Hello']" >&2; exit 1)
	PROMPT="$(PROMPT)" ./scripts/llama-infer.sh "$(MODEL)"

collect-env:
	./scripts/collect-env.sh

toolbox:
	./scripts/create-toolbox.sh

stop:
	./scripts/stop-container.sh

rm:
	./scripts/remove-container.sh

clean-runtime:
	./scripts/clean-runtime-artifacts.sh
