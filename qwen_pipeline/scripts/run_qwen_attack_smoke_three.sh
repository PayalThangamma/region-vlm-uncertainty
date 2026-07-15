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

rm -rf qwen_pipeline/outputs/attack_smoke_qwen
mkdir -p qwen_pipeline/outputs/attack_smoke_qwen

python qwen_pipeline/code/08_qwen_attack_smoke_three.py
