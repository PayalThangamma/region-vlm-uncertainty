#!/bin/bash
set -euo pipefail

cd /nethome/ptasathish/projects/region-vlm-uncertainty

source .venv_epistemic/bin/activate

export TMPDIR=/scratch/ptasathish/qwen_tmp
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export HF_HOME=/scratch/ptasathish/qwen_huggingface
export XDG_CACHE_HOME=/scratch/ptasathish/qwen_cache
export PYTHONPATH=/scratch/ptasathish/qwen_python_packages:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p qwen_pipeline/outputs/eval_removed_full

python \
  qwen_pipeline/code/14_qwen_masking_generation_full.py
