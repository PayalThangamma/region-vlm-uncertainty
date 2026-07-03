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

echo "Run 015A - Full eval removed images, region_mask_mode=all"
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
test -f hpc_inputs/questions_removed_jpg.jsonl
test -d hpc_inputs/removed_images_jpg
test -d outputs/attack_removed_full
test -d hpc_inputs/token_regions

echo "Question lines:"
wc -l hpc_inputs/questions_removed_jpg.jsonl

echo "Removed images:"
find hpc_inputs/removed_images_jpg -maxdepth 1 -type f | wc -l

echo "Attack images:"
find outputs/attack_removed_full -maxdepth 1 -type f | wc -l

echo "Token regions:"
find hpc_inputs/token_regions -mindepth 1 -maxdepth 1 -type d | wc -l

OUTDIR="outputs/eval_removed_all"

if [ -d "$OUTDIR" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
  echo "Existing $OUTDIR found. Moving to ${OUTDIR}_backup_${TS}"
  mv "$OUTDIR" "${OUTDIR}_backup_${TS}"
fi

cd Epistemic/baselines

export PYTHONPATH=/nethome/ptasathish/projects/region-vlm-uncertainty/Epistemic/baselines:${PYTHONPATH:-}

echo "Running full eval"
../../.venv_epistemic/bin/python eval_scripts/eval_caption.py \
  --model llava-1.5-7b \
  --decoder greedy \
  --dataset_name rohe \
  --image_folder ../../hpc_inputs/removed_images_jpg \
  --caption_file_path ../../hpc_inputs/questions_removed_jpg.jsonl \
  --attack_image_folder ../../outputs/attack_removed_full \
  --output_dir ../../outputs/eval_removed_all \
  --num_samples 522 \
  --max_new_tokens 64 \
  --use_ours \
  --region_mask_mode all \
  --token_region_root ../../hpc_inputs/token_regions

cd /nethome/ptasathish/projects/region-vlm-uncertainty

echo "Checking outputs"
find outputs/eval_removed_all -maxdepth 3 -type f | sort | head -20
wc -l outputs/eval_removed_all/captions.jsonl
find outputs/eval_removed_all/region_uncertainty -maxdepth 1 -type f -name "*.json" | wc -l

echo "Status: Run 015A full eval removed all completed."
date
