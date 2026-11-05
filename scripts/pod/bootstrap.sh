#!/bin/bash
#
# Bootstrap a fresh Runpod pod for the extraction pipeline. Idempotent.
#
# Assumes a CUDA PyTorch base image with the network volume mounted at
# /workspace. Clones the assistant-axis fork (master, which has the judge's
# --judge_extra_body) and installs it with the gpu extra (vllm).
#
# Layout rule: code and the venv live on the LOCAL container disk, because
# /workspace is a network filesystem and importing torch/vllm from it costs
# minutes per process start. Only what must survive the pod lives on
# /workspace: HF weight cache, vLLM compile cache, run outputs.
# Create pods with --container-disk-in-gb 40 so the venv (~12GB) fits.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
CODE_DIR="${CODE_DIR:-/opt/assistant-axis}"
FORK_URL="https://github.com/xiaohan2012/assistant-axis.git"
BRANCH="${FORK_BRANCH:-master}"

apt-get update -qq && apt-get install -y -qq tmux git || true

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d "$CODE_DIR" ]; then
    git clone "$FORK_URL" "$CODE_DIR"
fi
cd "$CODE_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only

# Persistent caches on the network volume: weights survive pod teardown,
# and the vLLM compile cache makes later engine starts skip compilation
export HF_HOME="$WORKSPACE/hf"
export VLLM_CACHE_ROOT="$WORKSPACE/vllm-cache"

uv sync --extra gpu

echo "bootstrap done: $(git rev-parse --short HEAD) on $BRANCH at $CODE_DIR"
