#!/bin/bash
set -euo pipefail

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}

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
mkdir -p outputs/smoke_eval_removed_all

source .venv_epistemic/bin/activate

echo "Run 014 — eval_caption.py smoke test"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

echo "Checking inputs"
ls hpc_inputs_smoke/questions_removed_jpg.jsonl
ls hpc_inputs_smoke/removed_images_jpg | head
ls outputs/smoke_attack_removed | head
ls hpc_inputs_smoke/token_regions | head

echo "Running eval_caption smoke test"

cd Epistemic/baselines

export PYTHONPATH=/nethome/ptasathish/projects/region-vlm-uncertainty/Epistemic/baselines:${PYTHONPATH:-}
echo "PYTHONPATH=$PYTHONPATH"

../../.venv_epistemic/bin/python eval_scripts/eval_caption.py \
  --model llava-1.5-7b \
  --decoder greedy \
  --dataset_name rohe \
  --image_folder ../../hpc_inputs_smoke/removed_images_jpg \
  --caption_file_path ../../hpc_inputs_smoke/questions_removed_jpg.jsonl \
  --attack_image_folder ../../outputs/smoke_attack_removed \
  --output_dir ../../outputs/smoke_eval_removed_all \
  --num_samples 3 \
  --max_new_tokens 64 \
  --use_ours \
  --region_mask_mode all \
  --token_region_root ../../hpc_inputs_smoke/token_regions

echo "Checking outputs"
cd /nethome/ptasathish/projects/region-vlm-uncertainty
find outputs/smoke_eval_removed_all -maxdepth 3 -type f | sort
wc -l outputs/smoke_eval_removed_all/captions.jsonl || true

echo "Status: eval_caption smoke test completed."
