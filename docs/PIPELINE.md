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

---

# Stage 15C - Full Removed-Image Evaluation with Removed-Region Masking

## Purpose

Evaluate LLaVA on all 522 ROHE removed-image samples while suppressing epistemically uncertain visual tokens only in the removed-object region.

This condition tests whether uncertainty in the location where the object was removed causally contributes to hallucination.

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

```text
522 / 522 samples completed
522 captions written
522 region_uncertainty JSON files written
runtime: about 40.7 minutes
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

It is compared against:

```text
Stage 15B: region_mask_mode = none
Stage 15A: region_mask_mode = all
Stage 15D: region_mask_mode = context
Stage 15E: region_mask_mode = background
```

In the final analysis, removed-region masking produced only a weak reduction in hallucination compared with global and background-region masking.

---

# Stage 15D - Full Removed-Image Evaluation with Context-Region Masking

## Purpose

Evaluate LLaVA on all 522 ROHE removed-image samples while suppressing epistemically uncertain visual tokens only in the context region around the removed object.

This condition tests whether hallucination is influenced by uncertain surrounding context rather than only by the removed-object region itself.

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

Other samples had non-zero suppressed tokens, confirming that context-region masking was applied.

## Interpretation

This is the context-region causal masking condition.

It is compared against:

```text
Stage 15B: region_mask_mode = none
Stage 15A: region_mask_mode = all
Stage 15C: region_mask_mode = removed
Stage 15E: region_mask_mode = background
```

In the final analysis, context-region masking produced only a small and statistically unreliable reduction in hallucination.

---

# Stage 15E - Full Removed-Image Evaluation with Background-Region Masking

## Purpose

Evaluate LLaVA on all 522 ROHE removed-image samples while suppressing epistemically uncertain visual tokens only in the background region.

This condition tests whether uncertain background tokens causally contribute to object hallucination. It also acts as an important comparison against removed-region and context-region masking.

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

```text
522 / 522 samples completed
522 captions written
522 region_uncertainty JSON files written
runtime: about 39.2 minutes
```

The output checks confirmed:

```text
522 outputs/eval_removed_background/captions.jsonl
522 region_uncertainty JSON files
```

Runtime:

```text
greedy 2352.300552368164 seconds
approximately 39.2 minutes
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

## Minor Logging Note

The log contained:

```text
sort: write failed: 'standard output': Broken pipe
sort: write error
```

This did not affect the evaluation outputs. It likely happened because a command similar to `find ... | sort | head -20` was used. When `head` closes the pipe after receiving enough lines, `sort` can report a broken pipe.

## Interpretation

This is the background-region causal masking condition.

In the final analysis, background-region masking produced the strongest region-specific reduction in hallucination.

This completed the five-condition removed-image experiment:

```text
Stage 15B: none
Stage 15A: all
Stage 15C: removed
Stage 15D: context
Stage 15E: background
```

---

# Stage 16 - Removed-Image Evaluation Analysis

**Status:** Completed  
**Dataset:** ROHE final removed-image evaluation outputs  
**Samples:** 522  
**Model:** LLaVA-1.5-7B  
**Conditions analyzed:** none, all, removed, context, background  

## Purpose

Analyze the completed removed-image evaluation outputs across all five masking conditions:

```text
none
all
removed
context
background
```

This stage computes hallucination rates, correct rejection rates, causal effects, object-category results, qualitative changed cases, and region uncertainty summaries.

## Input

```text
outputs/eval_removed_none/
outputs/eval_removed_all/
outputs/eval_removed_removed/
outputs/eval_removed_context/
outputs/eval_removed_background/
```

Each condition contains:

```text
captions.jsonl
config.json
region_uncertainty/*.json
```

Each condition had:

```text
522 captions
522 region_uncertainty JSON files
```

## Script

```text
code/08_analyze_removed_eval.py
```

## Output

```text
outputs/metrics/removed_eval_all_captions_long.csv
outputs/metrics/removed_eval_region_uncertainty_long.csv
outputs/metrics/removed_eval_per_sample.csv
outputs/metrics/removed_eval_summary.csv
outputs/metrics/removed_eval_by_category.csv
outputs/metrics/removed_eval_region_summary.csv

outputs/plots/hallucination_rate_by_condition.png
outputs/plots/causal_effect_by_condition.png
outputs/plots/mean_suppression_density_by_region.png
```

## Hallucination Rule

All removed-image samples have:

```text
label = no
```

The answer was classified as:

```text
hallucinated        if answer starts with "Yes"
correct rejection   if answer starts with "No"
unclear             otherwise
```

No unclear answers occurred in this run.

## Main Summary

```text
condition     n     hallucinated_yes   correct_rejection_no   hallucination_rate
none          522   415                107                    0.795019
all           522   396                126                    0.758621
removed       522   409                113                    0.783525
context       522   412                110                    0.789272
background    522   399                123                    0.764368
```

## Causal Effects

The causal effect was computed as:

```text
hallucination_rate_none - hallucination_rate_condition
```

Results:

```text
all          0.036398   = 3.64 percentage points
removed      0.011494   = 1.15 percentage points
context      0.005747   = 0.57 percentage points
background   0.030651   = 3.07 percentage points
```

## Interpretation

The no-masking baseline hallucinated the removed object in:

```text
415 / 522 samples
79.50%
```

Global uncertain-token masking reduced hallucination to:

```text
396 / 522 samples
75.86%
```

Among region-specific conditions, background-region masking was strongest:

```text
399 / 522 samples
76.44%
```

Removed-region and context-region masking produced much smaller reductions.

This suggests that hallucination is not mainly driven by uncertainty localized only to the removed-object region. Instead, the stronger effect of global and background masking suggests that hallucination may depend on broader scene-level visual uncertainty.

## Category-Level Findings

The masking effect was object-dependent.

The strongest improvements appeared for:

```text
bicycle
bench
```

For bicycle:

```text
none        21 / 26 hallucinated = 80.8%
all         17 / 26 hallucinated = 65.4%
background 17 / 26 hallucinated = 65.4%
```

For bench:

```text
none        60 / 66 hallucinated = 90.9%
all         53 / 66 hallucinated = 80.3%
background 55 / 66 hallucinated = 83.3%
```

Some categories showed little or no improvement.

Bottle was unchanged:

```text
none        23 / 29 hallucinated = 79.3%
all         23 / 29 hallucinated = 79.3%
removed     23 / 29 hallucinated = 79.3%
context     23 / 29 hallucinated = 79.3%
background  23 / 29 hallucinated = 79.3%
```

Dog was almost unchanged:

```text
none        54 / 73 hallucinated = 74.0%
background 53 / 73 hallucinated = 72.6%
```

## Qualitative Changed Cases

There were:

```text
32 changed cases
```

where the no-masking baseline hallucinated but at least one masking condition corrected the response.

Examples:

```text
sample_000148 - bicycle
none: Yes, there is a bicycle in the image.
removed/context/background: No, there is no bicycle in the image.

sample_000185 - bus
none: Yes, there is a bus in the image.
all/removed/context/background: No, there is no bus in the image.

sample_000205 - cat
none: Yes, there is a cat in the image.
all/removed/context/background: No, there is no cat in the image.

sample_000416 - chair
none: Yes, there is a chair in the image.
all/context/background: No, there is no chair in the image.
```

These examples support the quantitative result that uncertainty masking can sometimes shift the model from hallucinating the removed object to correctly rejecting it.

## Conclusion

Stage 16 produced the first complete result of the project.

The main finding is that global and background-region uncertain-token masking reduce object hallucination more than removed-region or context-region masking.

This supports a distributed scene-level interpretation of object hallucination in the removed-object setting.

---

# Stage 16B - Bootstrap Significance Analysis

**Status:** Completed  
**Bootstrap iterations:** 10000  
**Samples:** 522  
**Random seed:** 42  

## Purpose

Test whether the observed causal effects are statistically reliable using paired bootstrap resampling over the 522 samples.

The comparison is paired because every sample appears in every condition.

## Input

```text
outputs/metrics/removed_eval_per_sample.csv
```

## Script

```text
code/09_bootstrap_removed_eval.py
```

## Output

```text
outputs/metrics/removed_eval_bootstrap_effects.csv
```

## Bootstrap Results

```text
condition     observed_effect   effect_pp   95% CI pp          approx p-value
all           0.036398          3.64        [1.53, 5.94]       0.0014
removed       0.011494          1.15        [0.00, 2.30]       0.0642
context       0.005747          0.57        [-0.57, 1.72]      0.3840
background    0.030651          3.07        [0.96, 5.17]       0.0036
```

## Interpretation

Global uncertain-token masking produced a statistically supported reduction in hallucination:

```text
effect = 3.64 percentage points
95% CI = [1.53, 5.94]
p ≈ 0.0014
```

Background-region masking also produced a statistically supported reduction:

```text
effect = 3.07 percentage points
95% CI = [0.96, 5.17]
p ≈ 0.0036
```

Removed-region masking showed only a weak or borderline trend:

```text
effect = 1.15 percentage points
95% CI = [0.00, 2.30]
p ≈ 0.0642
```

Context-region masking was not statistically reliable:

```text
effect = 0.57 percentage points
95% CI = [-0.57, 1.72]
p ≈ 0.3840
```

## Conclusion

The bootstrap analysis supports the claim that global uncertain-token masking and background-region masking reduce hallucination compared with the no-masking baseline.

The results do not support a strong claim that removed-region or context-region uncertainty alone is the main causal driver.

The final interpretation is:

```text
Hallucination in this removed-object setting is not mainly explained by uncertainty localized to the removed-object region. Instead, hallucination appears to be more distributed, with background-region uncertainty showing a stronger causal effect than removed-region or context-region uncertainty.
```

## Final Result Statement

The no-masking baseline hallucinated the removed object in 79.50% of samples.

Global uncertain-token masking significantly reduced hallucination to 75.86%, with:

```text
effect = 3.64 percentage points
95% CI = [1.53, 5.94]
p ≈ 0.0014
```

Background-region masking significantly reduced hallucination to 76.44%, with:

```text
effect = 3.07 percentage points
95% CI = [0.96, 5.17]
p ≈ 0.0036
```

Removed-region masking showed only a weak trend, while context-region masking was not statistically reliable.

These results suggest that object hallucination after object removal is driven more by distributed scene-level uncertainty than by uncertainty localized only to the removed-object region.

---

# Different-Family Backend Attempt - Shikra

**Status:** Attempted, not used for final evaluation  

## Goal

Shikra was tested as a different-family VLM backend to check whether the epistemic region-wise masking method could be extended beyond LLaVA.

## Setup Completed

The infrastructure setup was successful:

- LLaMA-7B base checkpoint was downloaded.
- Shikra delta was applied.
- The merged checkpoint was created as `shikra-combined`.
- The merged checkpoint was linked into the Epistemic repo.
- The checkpoint loaded successfully on a V100 GPU.
- The smoke evaluation ran.
- The script wrote output files.

## Compatibility Fixes Attempted

Several compatibility fixes were required:

- Fixed `PYTHONPATH` for `minigpt4` imports.
- Removed unsupported generation kwargs passed to Hugging Face `generate`.
- Added compatibility defaults for missing decoding variables.
- Tested the default yes/no prompt.
- Tested a simple caption prompt: `Describe the image.`
- Tested prompt formatting with and without the system-message prefix.
- Tested newline-separated image/question prompt formatting.
- Added temporary debug logging to inspect generated token IDs.

## Result

Although Shikra loaded and ran, generation was degenerate.

For a simple caption prompt:

```text
Describe the image.
```

the generated token IDs were repeatedly:

```text
29900, 29900, 29900, ...
```

These decoded to:

```text
0000000000000000
```

This means the model was genuinely generating repeated `"0"` tokens.

The issue was not caused by:

- JSON output writing
- output slicing
- missing images
- Condor failure
- data-path failure

## Interpretation

The issue appears to be checkpoint/tokenizer/prompt/backend compatibility.

Because the model does not produce valid natural-language answers, running the full region-wise evaluation would produce meaningless results.

## Decision

Shikra is not used for the full evaluation.

The completed LLaVA-1.5-7B experiment remains the main result.

The recommended robustness run is:

```text
LLaVA-1.5-13B
```

Qwen can be considered as a future model-specific backend extension.

---

# Model Backend Protocol

## Purpose

This project compares VLMs using a shared epistemic-causal evaluation protocol.

The project does not require identical internal code paths across all models. It requires the same experimental logic.

## Shared Protocol

For each model, the protocol is:

1. Use the same ROHE removed-object samples.
2. Use the same removed images and questions.
3. Use the same semantic region maps.
4. Identify uncertain visual tokens within that model.
5. Assign uncertain visual tokens to semantic regions.
6. Apply the same masking conditions:
   - none
   - all
   - removed
   - context
   - background
7. Compute hallucination rate.
8. Report causal effect relative to the model's own no-mask baseline.

## Causal Effect

For each condition:

```text
causal_effect(region) =
hallucination_rate_none - hallucination_rate_region_masked
```

A positive value means that masking uncertain tokens in that condition reduced hallucination.

## Cross-Model Comparability

Different VLMs expose visual tokens differently.

For example:

```text
LLaVA visual tokens are not the same as Qwen visual tokens.
LLaVA vision encoder is not the same as Qwen vision encoder.
LLaVA image-token mapping is not the same as Qwen image-token mapping.
```

Therefore, raw token counts should not be directly compared across models.

Do not compare:

```text
LLaVA has X uncertain tokens and Qwen has Y uncertain tokens.
```

Instead, compare within-model causal effects:

```text
Does masking background-region uncertain tokens reduce hallucination in each model?
```

## Backend-Specific Implementation

The methodology is shared, but token extraction and masking are model-specific backend operations.

A model backend must define:

- how the model is loaded
- how images are preprocessed
- how prompts are formatted
- where visual tokens appear
- how visual-token uncertainty is estimated
- how token indices map to image regions
- how selected visual tokens are masked
- how outputs are decoded and scored

## Current Backend Status

| Model | Status | Project Use |
|---|---|---|
| LLaVA-1.5-7B | Working | Main completed experiment |
| LLaVA-1.5-13B | Supported by same structure | Recommended robustness run |
| MiniGPT-4 | Ran but produced unusable empty outputs | Not used |
| Shikra | Loaded but generated repeated `0` outputs | Not used |
| Qwen | Not plug-and-play | Possible future backend extension |

## Recommended Framing

The final project should be described as:

```text
We use a shared epistemic-causal evaluation protocol. Token extraction and masking are implemented through model-specific backends because VLM architectures expose visual tokens differently. Cross-model comparison is therefore based on within-model causal effects rather than raw token counts.
```
