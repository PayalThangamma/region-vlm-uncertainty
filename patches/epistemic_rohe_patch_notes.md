# Epistemic ROHE Patch Notes

The external `Epistemic/` repository is ignored by Git.

Local patches are applied to adapt the Epistemic codebase to the ROHE-style region-wise hallucination project.

## Files to patch

```text
Epistemic/baselines/attack.py
Epistemic/baselines/eval_scripts/eval_caption.py
```

## Patch 001 — `attack.py` ROHE Dataset Support

**Date:** 2026-06-30  
**File:** `Epistemic/baselines/attack.py`

### Goal

Adapt the Epistemic attack script so it can generate adversarial images for the ROHE dataset.

### Changes Made

- Added `rohe` as a supported dataset.
- Added ROHE image loading for `.jpg`, `.jpeg`, and `.png` files.
- Removed the 500-image limit for ROHE.
- Added error handling if no ROHE images are found.
- Replaced fragile `.replace(".jpg", ".png")` output naming with `os.path.splitext(...)`.
- Added `rohe` to both EVA and CLIP parser dataset choices.

### Verification

Verified with:

```powershell
Select-String -Path .\Epistemic\baselines\attack.py -Pattern "rohe","os.path.splitext","No images found"
```

## Patch 002 — `eval_caption.py` ROHE Dataset and Region Masking Support

**Date:** 2026-06-30  
**File:** `Epistemic/baselines/eval_scripts/eval_caption.py`

### Goal

Adapt the Epistemic evaluation script so it can run on the ROHE dataset and support full-dataset region-wise uncertainty masking.

### Changes Made

- Added `rohe` as a supported dataset.
- Added ROHE JSONL loading from:
  - `questions_original.jsonl`
  - `questions_removed.jsonl`
  - `questions_removed_jpg.jsonl`
- Added `sample_id` support.
- Added `label`, `target_object`, and `split` fields to answer outputs.
- Added `--token_region_root` for full-dataset region-map loading.
- Kept `--token_region_path` for single-sample debugging.
- Added `region_mask_mode="none"` for baseline/no-masking runs.
- Fixed adversarial image path construction using `os.path.splitext`.
- Added CLS-token-safe masking:
  - If model output has 577 tokens and region map has 576 tokens, the first token is treated as CLS.
  - CLS token is always kept.
  - Region masking is applied only to patch tokens.
- Saved per-sample region uncertainty statistics in:
  - `output_dir/region_uncertainty/sample_xxxxxx.json`

### Supported Region Mask Modes

```text
none
all
removed
context
background