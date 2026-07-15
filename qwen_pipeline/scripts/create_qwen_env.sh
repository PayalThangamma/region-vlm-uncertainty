#!/bin/bash
set -euo pipefail

PROJECT=/nethome/ptasathish/projects/region-vlm-uncertainty
ENV=/scratch/ptasathish/qwen_env/.venv_qwen

export TMPDIR=/scratch/ptasathish/qwen_tmp
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PIP_CACHE_DIR=/scratch/ptasathish/qwen_pip_cache
export HF_HOME=/scratch/ptasathish/qwen_huggingface
export XDG_CACHE_HOME=/scratch/ptasathish/qwen_cache

mkdir -p \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  "$(dirname "$ENV")"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python3: $(command -v python3)"
python3 --version

rm -rf "$ENV"

echo "Creating virtual environment"
python3 -m venv "$ENV"

echo "Activating virtual environment"
source "$ENV/bin/activate"

echo "Environment Python: $(which python)"
python --version

echo "Upgrading pip"
python -m pip install --no-cache-dir --upgrade pip setuptools wheel

echo "Installing PyTorch CUDA 11.8 build"
python -m pip install --no-cache-dir \
  torch==2.6.0 \
  torchvision \
  --index-url https://download.pytorch.org/whl/cu118

echo "Installing Qwen dependencies"
python -m pip install --no-cache-dir \
  transformers \
  accelerate \
  qwen-vl-utils \
  pillow \
  pandas \
  numpy \
  safetensors \
  sentencepiece

echo "Verifying imports"
python - <<'PY'
import torch
import transformers
import qwen_vl_utils
import accelerate

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Qwen environment installation completed")
PY

echo "Completed: $(date)"
