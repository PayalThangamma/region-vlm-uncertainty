## Debug 001 — Permission Denied in `token_regions/` After HPC Upload

**Date:** 2026-06-30  
**Stage:** Stage 09 — Upload and Verify HPC Inputs  
**Location:** HPC cluster

### Problem

After uploading and extracting `outputs/hpc_inputs.zip` on the HPC cluster, the image folders were accessible and had the correct counts:

```text
original_images: 522
removed_images: 522
removed_images_jpg: 522
````

However, checking `token_regions/` produced many permission errors:

```text
find: ‘hpc_inputs/token_regions/sample_xxxxxx’: Permission denied
```

The token-region directory count initially returned:

```text
0
```

### Cause

The extracted `token_regions/` folders had incorrect permissions after transfer/extraction.

### Fix

The permissions were fixed using:

```bash
chmod -R u+rwX hpc_inputs
```

### Verification

After applying the permission fix, the folder and file counts were checked again.

Expected verified counts:

```text
token_regions: 522
token_to_region.json: 522
region_counts.json: 522
metadata.json: 522
```

### Status

Resolved.

The HPC input package is now accessible and ready for LLaVA baseline evaluation.


## Debug 002 - eval_caption.py Compatibility Fixes

### Status:
Resolved. Smoke test completed successfully.

### Problem:
The patched Epistemic/LLaVA code was incompatible with the current environment:
- torch 2.6.0+cu118
- transformers 4.57.6

The failures were not caused by the ROHE dataset. They were caused by old LLaVA/Epistemic assumptions conflicting with newer Transformers internals.

### Resolved issues:

1. Unsupported custom kwargs were being forwarded into Hugging Face `generate()`.
   - Fix: removed unsupported direct generation arguments from `minigpt4/models/llava.py`.

2. LLaVA generation required an explicit attention mask.
   - Fix: added `attention_mask=torch.ones_like(input_ids)` in the generate call.

3. Cached generation caused `past_key_values` / `attention_mask` compatibility errors.
   - Fix: set `use_cache=False` in `minigpt4/models/llava.py`.

4. `llava_arch.py` expected `attention_mask` to exist during generation.
   - Fix: patched `prepare_inputs_labels_for_multimodal()` to create the attention mask from `input_ids` when needed.

5. `CLIPAttention` no longer had private helper `self._shape` in Transformers 4.57.6.
   - Fix: added local `_shape_clip()` fallback inside `forward_sclip` in `eval_caption.py`.

6. LLaMA decoder self-attention rejected unsupported `labels` argument.
   - Fix: removed `labels=labels` from the `self.self_attn(...)` call in `modeling_llama.py`.

7. LLaMA decoder self-attention rejected unsupported `head_list` argument.
   - Fix: removed `head_list=head_list` from the `self.self_attn(...)` call in `modeling_llama.py`.

### Validation:
After these fixes, Run 014 completed successfully:
- 3 / 3 smoke samples processed.
- `captions.jsonl` created.
- `region_uncertainty` JSON files created for all 3 samples.

### Working backup on HPC:
- `backups/run14_WORKING_eval_caption_smoke_20260702_220159/`
- `backups/run14_WORKING_eval_caption_smoke_code.zip`
- SHA256: `a88f26f50addba46f62da9a469b709932ff1c1d2b0ee4b480d6a9ec2ce6386bd`

### Conclusion:
The `eval_caption.py` path is now validated for a 3-sample ROHE smoke test. Do not modify the working model files without creating a new backup first.
