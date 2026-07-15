#!/bin/bash
set -euo pipefail

cd /nethome/ptasathish/projects/region-vlm-uncertainty

source .venv_epistemic/bin/activate

export TMPDIR=/scratch/ptasathish/qwen_tmp
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

export HF_HOME=/scratch/ptasathish/qwen_huggingface
export TRANSFORMERS_CACHE=/scratch/ptasathish/qwen_huggingface
export XDG_CACHE_HOME=/scratch/ptasathish/qwen_cache

export PYTHONPATH=/scratch/ptasathish/qwen_python_packages:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$TMPDIR" \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  qwen_pipeline/outputs

echo "Qwen baseline smoke test"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Python: $(which python)"

python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory GB:",
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
    )
PY

python qwen_pipeline/code/00_qwen_baseline_smoke.py \
  --questions hpc_inputs_smoke/questions_removed_jpg.jsonl \
  --image-root hpc_inputs_smoke/removed_images_jpg \
  --output qwen_pipeline/outputs/baseline_removed_smoke.jsonl \
  --num-samples 3 \
  --max-new-tokens 32

wc -l qwen_pipeline/outputs/baseline_removed_smoke.jsonl

echo "Qwen baseline smoke completed."
