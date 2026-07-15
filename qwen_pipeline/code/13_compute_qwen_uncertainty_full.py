import json
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

QUESTIONS = Path(
    "hpc_inputs/questions_removed_jpg.jsonl"
)

IMAGE_ROOT = Path(
    "hpc_inputs/removed_images_jpg"
)

# Geometry-aligned attacks are only for structural validation.
ATTACK_ROOT = Path(
    "qwen_pipeline/outputs/attack_removed_full"
)

REGION_ROOT = Path(
    "qwen_pipeline/outputs/token_regions_full"
)

OUTPUT_ROOT = Path(
    "qwen_pipeline/outputs/uncertainty_removed_full"
)

LAYER_INDICES = [0, 8, 16, 24, 31]
K_SIG = 1.1


class InputHook:
    def __init__(self, module):
        self.input_tensor = None
        self.handle = module.register_forward_hook(
            self._hook
        )

    def _hook(self, module, inputs, output):
        if not inputs:
            raise RuntimeError("Hook received no inputs.")

        tensor = inputs[0]

        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Expected tensor input, found {type(tensor)}"
            )

        self.input_tensor = tensor.detach()

    def close(self):
        self.handle.remove()


def read_rows(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))


    return rows


def build_inputs(
    processor,
    image_path: Path,
    question: str,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    return processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def merge_2x2_scores(
    scores: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_size: int,
) -> torch.Tensor:
    expected = grid_h * grid_w

    if scores.numel() != expected:
        raise ValueError(
            f"Expected {expected} pre-merge scores, "
            f"found {scores.numel()}"
        )

    if grid_h % merge_size != 0:
        raise ValueError(
            f"Grid height {grid_h} not divisible by {merge_size}"
        )

    if grid_w % merge_size != 0:
        raise ValueError(
            f"Grid width {grid_w} not divisible by {merge_size}"
        )

    score_grid = scores.reshape(grid_h, grid_w)

    merged = (
        score_grid
        .reshape(
            grid_h // merge_size,
            merge_size,
            grid_w // merge_size,
            merge_size,
        )
        .mean(dim=(1, 3))
    )

    return merged.reshape(-1)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    merge_size = int(
        processor.image_processor.merge_size
    )

    hooks = {
        index: InputHook(
            model.visual.blocks[index]
        )
        for index in LAYER_INDICES
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_rows(QUESTIONS)
    print("Samples to process:", len(rows))

    completed = 0
    skipped = 0

    for row in rows:
        sample_id = row["sample_id"]

        output_path = (
            OUTPUT_ROOT
            / f"{sample_id}.json"
        )

        if output_path.exists():
            print(sample_id, "already completed; skipping")
            skipped += 1
            continue

        question = row.get(
            "text",
            row.get("prompt"),
        )

        clean_path = IMAGE_ROOT / row["image"]
        attack_path = (
            ATTACK_ROOT
            / f"{Path(row['image']).stem}.pt"
        )

        clean_inputs = build_inputs(
            processor,
            clean_path,
            question,
        )

        attack_record = torch.load(
            attack_path,
            map_location="cpu",
        )

        clean_grid = clean_inputs[
            "image_grid_thw"
        ][0].tolist()

        attack_grid = attack_record[
            "image_grid_thw"
        ][0].tolist()

        if clean_grid != attack_grid:
            raise ValueError(
                f"{sample_id}: grid mismatch: "
                f"{clean_grid} vs {attack_grid}"
            )

        temporal, grid_h, grid_w = clean_grid

        if temporal != 1:
            raise ValueError(
                f"{sample_id}: expected temporal grid 1"
            )

        clean_inputs = {
            key: value.to(model.device)
            if hasattr(value, "to")
            else value
            for key, value in clean_inputs.items()
        }

        attacked_pixels = attack_record[
            "pixel_values"
        ].to(
            model.device,
            dtype=torch.float16,
        )

        attack_grid_tensor = attack_record[
            "image_grid_thw"
        ].to(model.device)

        with torch.inference_mode():
            _ = model(
                **clean_inputs,
                use_cache=False,
                return_dict=True,
            )

        clean_states = {
            index: hooks[index]
            .input_tensor
            .detach()
            .float()
            .cpu()
            for index in LAYER_INDICES
        }

        with torch.inference_mode():
            _ = model.visual(
                attacked_pixels,
                grid_thw=attack_grid_tensor,
            )

        attack_states = {
            index: hooks[index]
            .input_tensor
            .detach()
            .float()
            .cpu()
            for index in LAYER_INDICES
        }

        premerge_uncertainty = torch.zeros(
            grid_h * grid_w,
            dtype=torch.float32,
        )

        for index in LAYER_INDICES:
            clean = clean_states[index]
            attack = attack_states[index]

            if clean.shape != attack.shape:
                raise ValueError(
                    f"{sample_id}, layer {index}: "
                    "clean/attack mismatch"
                )

            if clean.ndim != 2:
                raise ValueError(
                    f"{sample_id}, layer {index}: "
                    f"expected 2D state, found {clean.shape}"
                )

            difference = attack - clean
            token_norm = difference.norm(dim=-1)

            max_value = token_norm.max()

            if max_value.item() > 0:
                token_norm = token_norm / max_value
            else:
                token_norm = torch.zeros_like(
                    token_norm
                )

            premerge_uncertainty += token_norm

        premerge_uncertainty /= len(
            LAYER_INDICES
        )

        merged_uncertainty = merge_2x2_scores(
            premerge_uncertainty,
            grid_h=grid_h,
            grid_w=grid_w,
            merge_size=merge_size,
        )

        token_map_path = (
            REGION_ROOT
            / sample_id
            / "token_to_region.json"
        )

        token_to_region = json.loads(
            token_map_path.read_text(
                encoding="utf-8"
            )
        )

        if len(token_to_region) != (
            merged_uncertainty.numel()
        ):
            raise ValueError(
                f"{sample_id}: token map has "
                f"{len(token_to_region)} entries, "
                f"uncertainty has "
                f"{merged_uncertainty.numel()}"
            )

        threshold = (
            merged_uncertainty.mean()
            + K_SIG
            * merged_uncertainty.std(
                unbiased=False
            )
        )

        uncertain = (
            merged_uncertainty >= threshold
        )

        region_stats = {}

        for region in [
            "removed",
            "context",
            "background",
        ]:
            region_indices = [
                int(token_id)
                for token_id, label
                in token_to_region.items()
                if label == region
            ]

            region_uncertain = sum(
                int(uncertain[index].item())
                for index in region_indices
            )

            region_stats[region] = {
                "total": len(region_indices),
                "uncertain": region_uncertain,
                "uncertainty_density": (
                    region_uncertain
                    / len(region_indices)
                    if region_indices
                    else 0.0
                ),
            }

        result = {
            "sample_id": sample_id,
            "image": row["image"],
            "image_grid_thw": clean_grid,
            "merge_size": merge_size,
            "premerge_token_count":
                int(premerge_uncertainty.numel()),
            "merged_token_count":
                int(merged_uncertainty.numel()),
            "layers": LAYER_INDICES,
            "k_sig": K_SIG,
            "threshold":
                float(threshold.item()),
            "mean_uncertainty":
                float(
                    merged_uncertainty.mean().item()
                ),
            "std_uncertainty":
                float(
                    merged_uncertainty.std(
                        unbiased=False
                    ).item()
                ),
            "num_uncertain_tokens":
                int(uncertain.sum().item()),
            "regions": region_stats,
            "merged_uncertainty": [
                float(value)
                for value in merged_uncertainty.tolist()
            ],
        }

        output_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        completed += 1

        print(
            sample_id,
            "premerge =",
            result["premerge_token_count"],
            "merged =",
            result["merged_token_count"],
            "uncertain =",
            result["num_uncertain_tokens"],
            "regions =",
            region_stats,
        )

    for hook in hooks.values():
        hook.close()

    print("\nQwen full uncertainty computation completed.")
    print("Processed this run:", completed)
    print("Skipped existing:", skipped)
    print("Total requested:", len(rows))


if __name__ == "__main__":
    main()
