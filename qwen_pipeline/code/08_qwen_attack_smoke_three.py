import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

QUESTIONS = Path(
    "hpc_inputs_smoke/questions_removed_jpg.jsonl"
)

IMAGE_ROOT = Path(
    "hpc_inputs_smoke/removed_images_jpg"
)

OUTPUT_ROOT = Path(
    "qwen_pipeline/outputs/attack_smoke_qwen"
)

NUM_SAMPLES = 3
STEPS = 5
EPSILON = 0.05
ALPHA = 0.01
BASE_SEED = 42


def read_rows(
    path: Path,
    limit: int,
) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

            if len(rows) >= limit:
                break

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
        return_tensors="pt",
    )


def run_attack(
    model,
    clean_pixels: torch.Tensor,
    grid_thw: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, list[dict], float]:
    with torch.no_grad():
        clean_visual = model.visual(
            clean_pixels,
            grid_thw=grid_thw,
        ).detach()

    generator = torch.Generator(
        device=clean_pixels.device
    )
    generator.manual_seed(seed)

    random_delta = torch.empty_like(
        clean_pixels
    ).uniform_(
        -EPSILON,
        EPSILON,
        generator=generator,
    )

    attacked = (
        clean_pixels.detach()
        + random_delta
    )
    attacked.requires_grad_(True)

    step_records = []

    for step in range(STEPS):
        attacked_visual = model.visual(
            attacked,
            grid_thw=grid_thw,
        )

        representation_difference = F.mse_loss(
            attacked_visual.float(),
            clean_visual.float(),
        )

        loss = -representation_difference

        if attacked.grad is not None:
            attacked.grad.zero_()

        loss.backward()

        if attacked.grad is None:
            raise RuntimeError(
                "No gradient produced for pixel_values."
            )

        gradient_norm = float(
            attacked.grad.float().norm().item()
        )

        with torch.no_grad():
            attacked -= (
                ALPHA
                * attacked.grad.sign()
            )

            perturbation = torch.clamp(
                attacked - clean_pixels,
                min=-EPSILON,
                max=EPSILON,
            )

            attacked.copy_(
                clean_pixels + perturbation
            )

        attacked = attacked.detach()
        attacked.requires_grad_(True)

        max_delta = float(
            (attacked - clean_pixels)
            .abs()
            .max()
            .item()
        )

        record = {
            "step": step + 1,
            "representation_mse":
                float(
                    representation_difference.item()
                ),
            "gradient_norm": gradient_norm,
            "max_abs_delta": max_delta,
        }

        step_records.append(record)

    with torch.no_grad():
        final_visual = model.visual(
            attacked,
            grid_thw=grid_thw,
        )

    final_difference = float(
        F.mse_loss(
            final_visual.float(),
            clean_visual.float(),
        ).item()
    )

    if not any(
        record["gradient_norm"] > 0
        for record in step_records
    ):
        raise RuntimeError(
            "Attack failed: all gradient norms are zero."
        )

    if final_difference <= step_records[0][
        "representation_mse"
    ]:
        raise RuntimeError(
            "Attack failed: representation difference "
            "did not increase."
        )

    return (
        attacked.detach(),
        step_records,
        final_difference,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    rows = read_rows(
        QUESTIONS,
        NUM_SAMPLES,
    )

    if len(rows) != NUM_SAMPLES:
        raise ValueError(
            f"Expected {NUM_SAMPLES} rows, "
            f"found {len(rows)}"
        )

    for sample_index, row in enumerate(rows):
        sample_id = row["sample_id"]
        image_path = IMAGE_ROOT / row["image"]
        question = row.get(
            "text",
            row.get("prompt"),
        )

        if question is None:
            raise KeyError(
                f"No question for {sample_id}"
            )

        inputs = build_inputs(
            processor,
            image_path,
            question,
        )

        clean_pixels = inputs[
            "pixel_values"
        ].to(
            model.device,
            dtype=torch.float16,
        )

        grid_thw = inputs[
            "image_grid_thw"
        ].to(model.device)

        seed = BASE_SEED + sample_index

        attacked, step_records, final_difference = (
            run_attack(
                model=model,
                clean_pixels=clean_pixels,
                grid_thw=grid_thw,
                seed=seed,
            )
        )

        output = {
            "sample_id": sample_id,
            "image": row["image"],
            "model_id": MODEL_ID,
            "seed": seed,
            "grid_thw":
                grid_thw.detach().cpu().tolist(),
            "pixel_values_shape":
                list(clean_pixels.shape),
            "steps": STEPS,
            "epsilon": EPSILON,
            "alpha": ALPHA,
            "final_representation_mse":
                final_difference,
            "step_records":
                step_records,
        }

        json_path = (
            OUTPUT_ROOT
            / f"{sample_id}.json"
        )

        tensor_path = (
            OUTPUT_ROOT
            / f"{sample_id}.pt"
        )

        json_path.write_text(
            json.dumps(output, indent=2),
            encoding="utf-8",
        )

        torch.save(
            {
                "sample_id": sample_id,
                "pixel_values":
                    attacked.cpu(),
                "image_grid_thw":
                    grid_thw.detach().cpu(),
                "seed": seed,
                "epsilon": EPSILON,
                "alpha": ALPHA,
                "steps": STEPS,
            },
            tensor_path,
        )

        print(
            sample_id,
            "grid =",
            output["grid_thw"],
            "shape =",
            output["pixel_values_shape"],
            "initial_mse =",
            step_records[0][
                "representation_mse"
            ],
            "final_mse =",
            final_difference,
            "max_delta =",
            max(
                record["max_abs_delta"]
                for record in step_records
            ),
        )

    print(
        "\nQwen three-sample attack smoke test passed."
    )


if __name__ == "__main__":
    main()
