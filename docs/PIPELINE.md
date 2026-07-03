# Stage 01 - Create Raw ROHE Dataset

**Script:** `code/01_create_rohe_dataset.py`  

## Objective

Generate a ROHE-style dataset from MS COCO validation images by selecting suitable object instances and creating the metadata and segmentation masks required for object removal.

## Why this stage is necessary

COCO contains original images and object annotations, but it does not contain paired images where an object has been removed. Therefore, the project first constructs its own object-removal dataset before inpainting.

## Inputs

```text
data/coco/val2017/
data/coco/annotations/instances_val2017.json
```

## Target Categories

```text
dog
cat
car
chair
bicycle
bus
bottle
bench
```

## Method

1. Load COCO annotations.
2. Select target object categories.
3. Find images containing one of the target categories.
4. Retrieve object instances.
5. Choose the largest object instance.
6. Keep only objects occupying 5% to 30% of the image.
7. Generate a binary segmentation mask with `annToMask()`.
8. Copy the original COCO image.
9. Generate `mask_overlay.png` for visual verification.
10. Store sample metadata.

## Output

```text
data/rohe_raw/
```

Each sample contains:

```text
original.jpg
mask.png
mask_overlay.png
metadata.json
```

Global output:

```text
data/rohe_raw/manifest.json
```

## Result

```text
636 candidate samples generated
```

---

# Stage 02 - Prepare LaMa Input

**Script:** `code/02_prepare_lama_input.py`  

## Objective

Convert the raw ROHE dataset into the flat input format expected by the LaMa inpainting model.

## Input

```text
data/rohe_raw/
```

## Output

```text
lama_input/
```

LaMa input files are named like:

```text
sample_xxxxxx.png
sample_xxxxxx_mask.png
```

## Result

```text
636 samples prepared for LaMa
```

---

# Stage 03 - LaMa Inpainting

## Objective

Generate object-removed images using the LaMa inpainting model.

## Input

```text
lama_input/
```

## Output

```text
lama_output/
```

## Environment

```text
conda activate lama39
```

## Execution

The setup and command are documented in:

```text
docs/LAMA_SETUP.md
```

## Result

```text
LaMa inpainting completed successfully
```

---

# Stage 04 - Integrate LaMa Outputs

**Script:** `code/03_copy_lama_output.py`  

## Objective

Copy the LaMa-generated images back into each ROHE sample directory as `removed.png`.

## Input

```text
lama_output/
data/rohe_raw/
```

## Output

```text
data/rohe_raw/sample_xxxxxx/removed.png
```

Each sample now contains:

```text
original.jpg
removed.png
mask.png
mask_overlay.png
metadata.json
```

## Result

```text
636 LaMa outputs copied back into data/rohe_raw/
```

---

# Stage 05 - Build Region Maps

**Script:** `code/04_build_region_maps.py`  

## Purpose

Generate semantic region maps for every raw ROHE sample. These maps are required for region-wise epistemic masking experiments.

Each image is divided into three semantic regions:

1. `removed` - the original COCO object mask
2. `context` - a dilated ring around the removed object
3. `background` - everything outside removed and context regions

## Input

```text
data/rohe_raw/
```

## Method

1. Load `mask.png`.
2. Resize the mask to `336 x 336`.
3. Use the resized mask as the removed-object region.
4. Create the context region by dilating the removed mask and subtracting the original removed mask.
5. Create the background region as all remaining pixels.
6. Divide the image into a `24 x 24` CLIP patch grid.
7. Assign each of the `576` patch tokens to `removed`, `context`, or `background`.

## Key Parameters

```text
IMAGE_SIZE = 336
PATCH_SIZE = 14
GRID_SIZE = 24
EXPECTED_TOKENS = 576
DILATION_SIZE = 35
OVERLAP_THRESHOLD = 0.25
```

## Output

```text
outputs/region_maps_rohe/
```

For each sample:

```text
removed_mask.png
context_mask.png
background_mask.png
token_to_region.json
region_counts.json
metadata.json
```

## Result

```text
Samples found: 636
Successful samples: 636
Skipped samples: 0
```

All successful samples were mapped to:

```text
24 x 24 = 576 patch tokens
```

---

# Stage 06 - Quality Filtering

**Script:** `code/05_quality_filter.py`  

## Purpose

Filter the raw ROHE samples using token-region statistics so the final dataset contains samples with usable removed-object, context, and background regions.

## Filtering Criteria

A sample is marked as good if:

```text
total == 576
background >= 250
20 <= removed <= 150
30 <= context <= 200
```

## Output

```text
outputs/rohe_quality.csv
```

## Result

```text
Samples checked: 636
Good samples: 522 / 636
Rejected samples: 114 / 636
```

---

# Stage 07 - Create Final Common Dataset

**Script:** `code/06_create_final_dataset.py`  
**Status:** Completed

## Purpose

Create the final common dataset used for all later experiments. This is needed so baseline, global masking, region-wise masking, and controls all run on the same set of samples.

## Input

```text
data/rohe_raw/
outputs/region_maps_rohe/
outputs/rohe_quality.csv
```

## Output

```text
data/rohe_final/
outputs/region_maps_final/
outputs/final_dataset_manifest.json
```

## Result

```text
Final samples: 522
```

---

# Stage 08 - Prepare HPC Inputs

**Script:** `code/07_prepare_hpc_inputs.py`  

## Purpose

Prepare the final ROHE dataset for running LLaVA and epistemic uncertainty experiments on the HPC cluster.

## Input

```text
data/rohe_final/
outputs/region_maps_final/
```

## Output

```text
outputs/hpc_inputs/
```

Generated folders:

```text
outputs/hpc_inputs/original_images/
outputs/hpc_inputs/removed_images/
outputs/hpc_inputs/removed_images_jpg/
outputs/hpc_inputs/token_regions/
```

Generated JSONL files:

```text
outputs/hpc_inputs/questions_original.jsonl
outputs/hpc_inputs/questions_removed.jsonl
outputs/hpc_inputs/questions_removed_jpg.jsonl
```

## Evaluation Meaning

For original images:

```text
label = yes
```

For removed images:

```text
label = no
```

## Result

```text
Prepared samples: 522
Skipped samples: 0
```

---

# Stage 09 - Upload and Verify HPC Inputs

## Purpose

Upload the prepared 522-sample HPC input package to the cluster and verify that all required files are accessible.

## HPC Location

```text
/nethome/ptasathish/projects/region-vlm-uncertainty/hpc_inputs/
```

## Verification

```text
original_images: 522
removed_images: 522
removed_images_jpg: 522
token_regions: 522
questions_original.jsonl: 522
questions_removed.jsonl: 522
questions_removed_jpg.jsonl: 522
```

## Result

The HPC input package was uploaded, extracted, permission-fixed, and verified.

---

# Stage 10 - Patch, Upload, and Verify Epistemic Code

## Purpose

Adapt the external Epistemic repository so it can run on the ROHE dataset and support region-wise masking experiments.

## Patched Files

```text
Epistemic/baselines/attack.py
Epistemic/baselines/eval_scripts/eval_caption.py
```

## Main Changes

`attack.py`:

- Added `rohe` as a supported dataset option.
- Added ROHE image loading for `.jpg`, `.jpeg`, and `.png` files.
- Removed the 500-image limit for ROHE.
- Added extension-safe output naming using `os.path.splitext`.

`eval_caption.py`:

- Added ROHE JSONL loading.
- Added `sample_id`, `label`, `target_object`, and `split` support.
- Added `--token_region_root`.
- Added `region_mask_mode = none`.
- Added CLS-token-safe masking logic.
- Saved per-sample region uncertainty JSON files.

## Result

The patched Epistemic code was uploaded and verified on HPC.

---

# Stage 11 - Prepare Smoke-Test Inputs

## Purpose

Create a 3-sample subset from the full HPC input package to test the patched attack and evaluation scripts before full runs.

## Smoke Samples

```text
sample_000062
sample_000068
sample_000069
```

## Output

```text
hpc_inputs_smoke/
```

## Result

```text
removed_images_jpg: 3
removed_images: 3
original_images: 3
token_regions: 3
questions_original.jsonl: 3
questions_removed.jsonl: 3
questions_removed_jpg.jsonl: 3
```

---

# Stage 12 - Attack Smoke Test

## Purpose

Verify that the patched Epistemic `attack.py` script can run on ROHE smoke-test data through Condor on a GPU node.

## Configuration

```text
Script: Epistemic/baselines/attack.py
Mode: clip
Model: openai/clip-vit-large-patch14-336
Dataset: rohe
Epsilon: 3
Alpha: 1
Steps: 2
```

## Output

```text
outputs/smoke_attack_removed/
```

Generated adversarial images:

```text
sample_000062.png
sample_000068.png
sample_000069.png
```

## Result

The ROHE adversarial attack pipeline was validated end-to-end on HPC.


# Stage 13 - Full Adversarial Attack Generation

## Purpose

Generate adversarial versions of all final ROHE removed images using the patched Epistemic CLIP attack pipeline.

These adversarial images are required for computing epistemic uncertainty by comparing clean removed images and adversarial removed images.

## Input

```text
hpc_inputs/removed_images_jpg/
```

## Output

```text
outputs/attack_removed_full/
```

## Configuration

```text
Script: Epistemic/baselines/attack.py
Mode: clip
Model: openai/clip-vit-large-patch14-336
Dataset: rohe
Epsilon: 3
Alpha: 1
Steps: 200
```

## Result

```text
522 adversarial images generated
Runtime: 1:50:53
```

---

# Stage 14 - eval_caption.py Smoke Test

## Purpose

Validate the patched `eval_caption.py` pipeline on 3 ROHE removed-image smoke samples before launching full LLaVA evaluation jobs.

## Configuration

```text
model: llava-1.5-7b
decoder: greedy
dataset_name: rohe
num_samples: 3
max_new_tokens: 64
use_ours: enabled
region_mask_mode: all
```

## Verified

- ROHE JSONL loading
- removed-image loading
- adversarial image loading
- token-region loading
- region-wise uncertain-token masking
- LLaVA generation
- `captions.jsonl` writing
- per-sample `region_uncertainty` JSON writing

## Result

```text
3 / 3 samples completed
```

---

# Stage 15A - Full Removed-Image Evaluation with Global Uncertain-Token Masking

## Purpose

Evaluate LLaVA on all 522 removed-image samples while suppressing all epistemically uncertain visual tokens, regardless of semantic region.

## Output

```text
outputs/eval_removed_all/
```

Generated files:

```text
captions.jsonl
config.json
region_uncertainty/*.json
```

## Configuration

```text
model: llava-1.5-7b
dataset: rohe
samples: 522
decoder: greedy
max_new_tokens: 64
region_mask_mode: all
```

## Result

```text
522 / 522 samples completed
522 captions written
522 region_uncertainty JSON files written
runtime: about 39.7 minutes
```

## Interpretation

This is the global uncertain-token masking condition. It will be compared against the no-masking baseline and region-specific masking conditions.

---

# Stage 15B - Full Removed-Image Evaluation Without Masking

## Purpose

Run the no-masking baseline on all 522 removed-image samples.

This condition is necessary because all global and region-wise masking conditions must be compared against an unmasked baseline.

## Output

```text
outputs/eval_removed_none/
```

Generated files:

```text
captions.jsonl
config.json
region_uncertainty/*.json
```

## Configuration

```text
model: llava-1.5-7b
dataset: rohe
samples: 522
decoder: greedy
max_new_tokens: 64
region_mask_mode: none
```

## Result

```text
522 / 522 samples completed
522 captions written
522 region_uncertainty JSON files written
runtime: about 40.0 minutes
```

## Interpretation

This is the baseline hallucination condition. It will be compared against `all`, `removed`, `context`, and `background` masking.

---

# Stage 15C - Full Removed-Image Evaluation with Removed-Region Masking

## Purpose

Evaluate LLaVA on all 522 removed-image samples while suppressing epistemically uncertain visual tokens only in the removed-object region.

This tests whether uncertainty in the location where the object was removed causally contributes to hallucination.

## Output

```text
outputs/eval_removed_removed/
```

Generated files:

```text
captions.jsonl
config.json
region_uncertainty/*.json
```

## Configuration

```text
model: llava-1.5-7b
dataset: rohe
samples: 522
decoder: greedy
max_new_tokens: 64
region_mask_mode: removed
```

## Result

```text
522 / 522 samples completed
522 captions written
522 region_uncertainty JSON files written
runtime: about 40.7 minutes
```

## Interpretation

This is the removed-region causal masking condition.

If this condition reduces hallucination more than context or background masking, it supports the hypothesis that uncertain tokens in the removed-object region are causally important.

If it has little effect, hallucination may be driven more by surrounding context, background cues, or language priors.

---

# Stage 15C — Full Removed-Image Evaluation with Removed-Region Masking

## Purpose

Evaluate LLaVA on all 522 removed-image samples while suppressing epistemically uncertain visual tokens only in the removed-object region.

This stage tests whether uncertainty in the location where the object was removed causally contributes to object hallucination.

This is the first region-specific causal masking condition.

## Input

```text
hpc_inputs/questions_removed_jpg.jsonl
hpc_inputs/removed_images_jpg/
outputs/attack_removed_full/
hpc_inputs/token_regions/
```

## Output

```text
outputs/eval_removed_removed/
```

Generated files:

```text
captions.jsonl
config.json
region_uncertainty/*.json
```

## Configuration

```text
model: llava-1.5-7b
dataset: rohe
samples: 522
decoder: greedy
max_new_tokens: 64
use_ours: enabled
region_mask_mode: removed
```

## Result

The run completed successfully.

```text
522 / 522 samples completed
522 captions written
522 region_uncertainty JSON files written
runtime: ~40.7 minutes
```

The output checks confirmed:

```text
522 outputs/eval_removed_removed/captions.jsonl
522 region_uncertainty JSON files
```

## Masking Meaning

This stage uses:

```text
region_mask_mode = removed
```

Therefore, uncertain visual tokens were suppressed only if they belonged to the removed-object region.

Uncertain tokens in the context and background regions were not suppressed in this condition.

## Sanity Check

The logs confirmed that removed-region masking was active:

```text
Region mode: removed
patch_total: 576
keep_mask_total: 577
```

Some samples had non-zero removed-region suppression, for example:

```text
suppressed_patch: 10
suppressed_patch: 13
suppressed_patch: 23
```

Some samples had:

```text
suppressed_patch: 0
```

This is expected. It means that, for those samples, no epistemically uncertain patch tokens fell inside the removed-object region.

## Interpretation

This is the removed-region causal masking condition.

It will be compared against:

```text
Stage 15B: region_mask_mode = none
Stage 15A: region_mask_mode = all
Stage 15D: region_mask_mode = context
Stage 15E: region_mask_mode = background
```

If removed-region masking reduces hallucination more than context or background masking, it supports the hypothesis that uncertain tokens in the removed-object region are causally important for hallucination.

If removed-region masking has little effect, hallucination may be driven more by surrounding context, background cues, or language priors.

## Result

Run 015C completed successfully and produced full-dataset outputs for all 522 removed-image samples with removed-region uncertain-token masking.

---

# Stage 15D — Full Removed-Image Evaluation with Context-Region Masking

## Purpose

Evaluate LLaVA on all 522 removed-image samples while suppressing epistemically uncertain visual tokens only in the context region around the removed object.

This stage tests whether hallucination is influenced by uncertain surrounding context rather than only by the removed-object region itself.

The context region is the dilated area around the removed object. It may contain co-occurring objects, scene cues, or background structures that can make the model infer that the removed object is still present.

## Input

```text
hpc_inputs/questions_removed_jpg.jsonl
hpc_inputs/removed_images_jpg/
outputs/attack_removed_full/
hpc_inputs/token_regions/
```

## Output

```text
outputs/eval_removed_context/
```

Generated files:

```text
captions.jsonl
config.json
region_uncertainty/*.json
```

## Configuration

```text
model: llava-1.5-7b
dataset: rohe
samples: 522
decoder: greedy
max_new_tokens: 64
use_ours: enabled
region_mask_mode: context
```

## Result

The run completed successfully.

```text
522 / 522 samples completed
522 captions written
522 region_uncertainty JSON files written
```

The output checks confirmed:

```text
522 outputs/eval_removed_context/captions.jsonl
522 region_uncertainty JSON files
```

## Masking Meaning

This stage uses:

```text
region_mask_mode = context
```

Therefore, uncertain visual tokens were suppressed only if they belonged to the context region.

Uncertain tokens in the removed-object region and background region were not suppressed in this condition.

## Sanity Check

The logs confirmed that context-region masking was active:

```text
Region mode: context
patch_total: 576
keep_mask_total: 577
```

Some samples may have:

```text
suppressed_patch: 0
```

This is expected. It means that, for those samples, no epistemically uncertain patch tokens fell inside the context region.

Other samples should have non-zero suppressed tokens, confirming that context-region masking was applied.

## Interpretation

This is the context-region causal masking condition.

It will be compared against:

```text
Stage 15B: region_mask_mode = none
Stage 15A: region_mask_mode = all
Stage 15C: region_mask_mode = removed
Stage 15E: region_mask_mode = background
```

If context-region masking reduces hallucination more than removed-region or background masking, it suggests that hallucination may be driven by surrounding contextual cues or object co-occurrence priors.

If context-region masking has little effect, it suggests that the surrounding context alone may not be the main cause of hallucination.

## Result

Run 015D completed successfully and produced full-dataset outputs for all 522 removed-image samples with context-region uncertain-token masking.



## Output

```text
outputs/eval_removed_context/
```

Generated files:

```text
captions.jsonl
config.json
region_uncertainty/*.json
```

## Configuration

```text
model: llava-1.5-7b
dataset: rohe
samples: 522
decoder: greedy
max_new_tokens: 64
use_ours: enabled
region_mask_mode: context
```

## Result

The run completed successfully.

```text
522 / 522 samples completed
522 captions written
522 region_uncertainty JSON files written
```

## Masking Meaning

This stage uses:

```text
region_mask_mode = context
```

Therefore, uncertain visual tokens were suppressed only if they belonged to the context region.

Uncertain tokens in the removed-object region and background region were not suppressed in this condition.

## Interpretation

This is the context-region causal masking condition.

It will be compared against:

```text
Stage 15B: region_mask_mode = none
Stage 15A: region_mask_mode = all
Stage 15C: region_mask_mode = removed
Stage 15E: region_mask_mode = background
```

If context-region masking reduces hallucination more than removed-region or background masking, it suggests that hallucination may be driven by surrounding contextual cues or object co-occurrence priors.

If context-region masking has little effect, it suggests that the surrounding context alone may not be the main cause of hallucination.

---

# Stage 15E — Full Removed-Image Evaluation with Background-Region Masking

## Purpose

Evaluate LLaVA on all 522 removed-image samples while suppressing epistemically uncertain visual tokens only in the background region.

This stage tests whether uncertain background tokens causally contribute to object hallucination.

It also acts as an important comparison against removed-region and context-region masking.

## Input

```text
hpc_inputs/questions_removed_jpg.jsonl
hpc_inputs/removed_images_jpg/
outputs/attack_removed_full/
hpc_inputs/token_regions/
```

## Output

```text
outputs/eval_removed_background/
```

Generated files:

```text
captions.jsonl
config.json
region_uncertainty/*.json
```

## Configuration

```text
model: llava-1.5-7b
dataset: rohe
samples: 522
decoder: greedy
max_new_tokens: 64
use_ours: enabled
region_mask_mode: background
```

## Result

The run completed successfully.

```text
522 / 522 samples completed
522 captions written
522 region_uncertainty JSON files written
runtime: ~39.2 minutes
```

The output checks confirmed:

```text
522 outputs/eval_removed_background/captions.jsonl
522 region_uncertainty JSON files
```

## Masking Meaning

This stage uses:

```text
region_mask_mode = background
```

Therefore, uncertain visual tokens were suppressed only if they belonged to the background region.

Uncertain tokens in the removed-object region and context region were not suppressed in this condition.

## Sanity Check

The logs confirmed that background-region masking was active:

```text
Region mode: background
patch_total: 576
keep_mask_total: 577
```

Many samples had non-zero background suppression, for example:

```text
suppressed_patch: 124
suppressed_patch: 110
suppressed_patch: 94
suppressed_patch: 93
suppressed_patch: 90
```

This confirms that background-region uncertain-token masking was applied.

## Interpretation

This is the background-region causal masking condition.

It will be compared against:

```text
Stage 15B: region_mask_mode = none
Stage 15A: region_mask_mode = all
Stage 15C: region_mask_mode = removed
Stage 15D: region_mask_mode = context
```

If background-region masking reduces hallucination strongly, it suggests that hallucination may be influenced by global scene/background cues.

If background-region masking has little effect compared with removed-region or context-region masking, it suggests that the hallucination is more localized to the removed-object region or nearby context.
