# Run 014 Compatibility Notes

Run 014 eval_caption.py smoke test completed successfully on 3 / 3 ROHE removed-image samples.

## Environment

- Python 3.9.25
- torch 2.6.0+cu118
- transformers 4.57.6
- CUDA available: True

## Outputs produced

- outputs/smoke_eval_removed_all/captions.jsonl
- outputs/smoke_eval_removed_all/config.json
- outputs/smoke_eval_removed_all/region_uncertainty/sample_000062.json
- outputs/smoke_eval_removed_all/region_uncertainty/sample_000068.json
- outputs/smoke_eval_removed_all/region_uncertainty/sample_000069.json

## Compatibility fixes applied on HPC

The external Epistemic/LLaVA repo is ignored by Git, so these changes are documented separately.

### minigpt4/models/llava.py
- Removed unsupported custom kwargs from the Hugging Face generate() call.
- Added attention_mask=torch.ones_like(input_ids).
- Set use_cache=False for Transformers 4.57 compatibility.

### minigpt4/models/llava_arch.py
- Patched attention-mask handling during generation to avoid NoneType crashes.

### eval_scripts/eval_caption.py
- Added _shape_clip fallback because CLIPAttention no longer exposes self._shape in Transformers 4.57.6.

### minigpt4/models/modeling_llama.py
- Removed labels=labels from self.self_attn(...).
- Removed head_list=head_list from self.self_attn(...).

## Working backup on HPC

- backups/run14_WORKING_eval_caption_smoke_20260702_220159/
- backups/run14_WORKING_eval_caption_smoke_code.zip
- SHA256: a88f26f50addba46f62da9a469b709932ff1c1d2b0ee4b480d6a9ec2ce6386bd

## Conclusion

Run 014 validates:
- ROHE JSONL loading
- removed-image loading
- adversarial image loading
- token-region loading
- region-wise uncertain-token masking
- LLaVA generation
- captions.jsonl writing
- region_uncertainty JSON writing
