ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_TAG=b8411
ARG LLAMA_CPP_COMMIT=4efd326e7107f3c6a84d23455ac620e7220079ec
ARG FLASH_ATTENTION_TAG=v2.8.4-cktile
ARG FLASH_ATTENTION_AITER_COMMIT=428e8e761c7bc22d03513bcb8507375afef1f916
ARG FLASH_ATTENTION_AITER_CK_COMMIT=b0c13f312443332c7c13a8cd26b3662582c8d3d4
ARG UNSLOTH_COMMIT=8b4a0f219127c4258fa2ea1adddd505223fdced9
ARG UNSLOTH_ZOO_COMMIT=378a9052b3e6ba972ca14341d11ec9fda64a311a

ENV VENV_PATH=/opt/venvs/strix \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/venvs/strix/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    MIOPEN_DISABLE_CACHE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    git-lfs \
    libcurl4-openssl-dev \
    libnuma-dev \
    ninja-build \
    pkg-config \
    procps \
    python3-dev \
    python3-venv \
    rsync \
    sudo \
    tini \
    unzip \
    vim \
    wget \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv ${VENV_PATH} \
  && ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel packaging \
  && ${VENV_PATH}/bin/pip install \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl" \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl" \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl" \
  && ${VENV_PATH}/bin/pip install \
    numpy==1.26.4 \
    jupyterlab==4.5.6 \
    notebook==7.5.0 \
    ipykernel==7.1.0 \
    ipywidgets==8.1.8 \
    matplotlib==3.10.7 \
    pandas==2.3.3 \
    scikit-learn==1.7.2 \
    datasets==4.1.1 \
    accelerate==1.10.1 \
    transformers==4.57.0 \
    peft==0.17.1 \
    trl==0.23.1 \
    sentencepiece==0.2.1 \
    safetensors==0.6.2 \
    huggingface-hub==0.35.1

RUN mkdir -p /opt/llama.cpp \
  && curl -fsSL "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/${LLAMA_CPP_COMMIT}" \
  | tar -xz --strip-components=1 -C /opt/llama.cpp \
  && cd /opt/llama.cpp \
  && cmake -S . -B build \
    -DGGML_HIP=ON \
    -DGGML_RPC=ON \
    -DLLAMA_HIP_UMA=ON \
    -DAMDGPU_TARGETS=gfx1151 \
    -DCMAKE_BUILD_TYPE=Release \
    -DROCM_PATH=/opt/rocm \
    -DHIP_PATH=/opt/rocm \
    -DHIP_PLATFORM=amd \
    -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm -mllvm --amdgpu-unroll-threshold-local=600" \
  && cmake --build build --config Release --parallel "$(nproc)" \
  && cmake --install build --config Release

RUN mkdir -p /tmp/flash-attention/third_party/aiter/3rdparty/composable_kernel \
  && curl -fsSL "https://codeload.github.com/ROCm/flash-attention/tar.gz/refs/tags/${FLASH_ATTENTION_TAG}" \
  | tar -xz --strip-components=1 -C /tmp/flash-attention \
  && curl -fsSL "https://codeload.github.com/ROCm/aiter/tar.gz/${FLASH_ATTENTION_AITER_COMMIT}" \
  | tar -xz --strip-components=1 -C /tmp/flash-attention/third_party/aiter \
  && curl -fsSL "https://codeload.github.com/ROCm/composable_kernel/tar.gz/${FLASH_ATTENTION_AITER_CK_COMMIT}" \
  | tar -xz --strip-components=1 -C /tmp/flash-attention/third_party/aiter/3rdparty/composable_kernel \
  && cd /tmp/flash-attention \
  && FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
     GPU_ARCHS=gfx1151 \
     PYTORCH_ROCM_ARCH=gfx1151 \
     TRITON_HIP_LLD_PATH=/opt/rocm/lib/llvm/bin/ld.lld \
     MAX_JOBS="$(nproc)" \
     ${VENV_PATH}/bin/pip install --no-build-isolation --no-deps .

RUN mkdir -p /tmp/unsloth-zoo /tmp/unsloth \
  && curl -fsSL "https://codeload.github.com/unslothai/unsloth-zoo/tar.gz/${UNSLOTH_ZOO_COMMIT}" \
  | tar -xz --strip-components=1 -C /tmp/unsloth-zoo \
  && ${VENV_PATH}/bin/pip install /tmp/unsloth-zoo \
  && curl -fsSL "https://codeload.github.com/unslothai/unsloth/tar.gz/${UNSLOTH_COMMIT}" \
  | tar -xz --strip-components=1 -C /tmp/unsloth \
  && cd /tmp/unsloth \
  && ${VENV_PATH}/bin/pip install ".[rocm72-torch291]"

RUN ${VENV_PATH}/bin/python -m ipykernel install --name strix-halo --display-name "Python (Strix Halo ROCm 7.2)" --prefix /usr/local

COPY scripts/container-start-notebook.sh /usr/local/bin/start-notebook
COPY scripts/container-smoke-test.sh /usr/local/bin/smoke-test
RUN chmod +x /usr/local/bin/start-notebook /usr/local/bin/smoke-test

RUN printf '%s\n' \
  'export VENV_PATH=/opt/venvs/strix' \
  'export ROCM_PATH=/opt/rocm' \
  'export HIP_PATH=/opt/rocm' \
  'export PATH=/opt/venvs/strix/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH' \
  'export LD_LIBRARY_PATH=/usr/local/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}' \
  'export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE' \
  'export GPU_ARCHS=gfx1151' \
  'export PYTORCH_ROCM_ARCH=gfx1151' \
  'export TRITON_HIP_LLD_PATH=/opt/rocm/lib/llvm/bin/ld.lld' \
  'export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1' \
  'export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1' \
  > /etc/profile.d/strix-halo-env.sh \
  && chmod +x /etc/profile.d/strix-halo-env.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-lc", "source /etc/profile.d/strix-halo-env.sh && sleep infinity"]
