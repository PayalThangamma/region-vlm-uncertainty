#!/usr/bin/env bash
set -euo pipefail

cd /nethome/ptasathish/projects/region-vlm-uncertainty

export TMPDIR=/scratch/ptasathish/tmp
export TEMP=/scratch/ptasathish/tmp
export TMP=/scratch/ptasathish/tmp
export PIP_CACHE_DIR=/scratch/ptasathish/pip_cache
export XDG_CACHE_HOME=/scratch/ptasathish/cache
export HF_HOME=/scratch/ptasathish/huggingface
export TRANSFORMERS_CACHE=/scratch/ptasathish/huggingface
export TORCH_HOME=/scratch/ptasathish/torch

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME"
mkdir -p logs outputs

source .venv_epistemic/bin/activate

echo "Run 015B - Full eval removed images, region_mask_mode=none"
date
hostname
which python
python -V

python - << 'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

echo "Checking full inputs"
test -f hpc_inputs/questions_original.jsonl
test -d hpc_inputs/original_images
test -d outputs/attack_original_full
test -d hpc_inputs/token_regions

echo "Question lines:"
wc -l hpc_inputs/questions_original.jsonl

echo "Removed images:"
find hpc_inputs/original_images -maxdepth 1 -type f | wc -l

echo "Attack images:"
find outputs/attack_original_full -maxdepth 1 -type f | wc -l

echo "Token regions:"
find hpc_inputs/token_regions -mindepth 1 -maxdepth 1 -type d | wc -l

OUTDIR="outputs/eval_original_none"

if [ -d "$OUTDIR" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
  echo "Existing $OUTDIR found. Moving to ${OUTDIR}_backup_${TS}"
  mv "$OUTDIR" "${OUTDIR}_backup_${TS}"
fi

cd Epistemic/baselines

export PYTHONPATH=/nethome/ptasathish/projects/region-vlm-uncertainty/Epistemic/baselines:${PYTHONPATH:-}

echo "Running full eval baseline none"
../../.venv_epistemic/bin/python eval_scripts/eval_caption.py \
  --model llava-1.5-7b \
  --decoder greedy \
  --dataset_name rohe \
  --image_folder ../../hpc_inputs/original_images \
  --caption_file_path ../../hpc_inputs/questions_original.jsonl \
  --attack_image_folder ../../outputs/attack_original_full \
  --output_dir ../../outputs/eval_original_none \
  --num_samples 522 \
  --max_new_tokens 64 \
  --use_ours \
  --region_mask_mode none \
  --token_region_root ../../hpc_inputs/token_regions

cd /nethome/ptasathish/projects/region-vlm-uncertainty

echo "Checking outputs"
find outputs/eval_original_none -maxdepth 3 -type f | sort | head -20
wc -l outputs/eval_original_none/captions.jsonl
find outputs/eval_original_none/region_uncertainty -maxdepth 1 -type f -name "*.json" | wc -l

echo "Status: Run 015B full eval removed none completed."
date
