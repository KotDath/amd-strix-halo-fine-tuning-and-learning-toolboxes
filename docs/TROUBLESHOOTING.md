# Troubleshooting

## `run-dev-container.sh` fails with IPC / shm errors

This repo intentionally uses `--ipc=host` and does not set `--shm-size`.
If you see a stale command line with both options, you are not running the
current script version.

## `llama.cpp` build fails during `git clone`

The current image build downloads `llama.cpp` and `unsloth` sources through
GitHub tarballs instead of `git clone`. If you still see `RPC failed` or
`HTTP/2 stream ... was not closed cleanly`, rebuild with the current
`Dockerfile`.

## PyTorch installs CUDA / NVIDIA wheels instead of ROCm wheels

The current image pins ROCm 7.2 wheels directly from `repo.radeon.com`. If you
see `nvidia-cuda-*` packages in the build log, you are building from an older
`Dockerfile`.

## Jupyter starts but is not reachable

Check:

- the port bind host in `.env` or the default `127.0.0.1`
- [logs/jupyter.log](/home/kotdath/omp/personal/strix-demo/amd-strix-halo-fine-tuning-and-learning-toolboxes/logs/jupyter.log)
- whether another process already owns the port

Then restart:

```bash
make stop
make run
make notebook
```

## `unsloth` fine-tuning hangs on the first run

The first run can spend a few minutes on:

- model download
- ROCm JIT compilation
- Unsloth kernel initialization

This is expected. Subsequent runs are much faster because the Hugging Face cache
and compiled kernels are reused.

## `unsloth` downloads are slow or unstable

The scripts default to:

- `HF_HUB_DISABLE_XET=1`
- `HF_HUB_ENABLE_HF_TRANSFER=0`

That avoids the flaky Xet path that caused partial downloads during testing.

## Need a full environment snapshot for debugging

Run:

```bash
make collect-env
```

The report is written under `reports/` and includes image/container inspect
data, `torch.utils.collect_env`, `llama.cpp` device enumeration, and ROCm
runtime information.
