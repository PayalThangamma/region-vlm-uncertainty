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
    "qwen_pipeline/outputs/attack_gradient_smoke"
)

STEPS = 5
EPSILON = 0.05
ALPHA = 0.01


def read_first_row(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(next(handle))


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

    # We only need gradients with respect to pixel_values.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Pin processor behavior explicitly.
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    row = read_first_row(QUESTIONS)

    sample_id = row["sample_id"]
    image_path = IMAGE_ROOT / row["image"]
    question = row.get(
        "text",
        row.get("prompt"),
    )

    inputs = build_inputs(
        processor,
        image_path,
        question,
    )

    clean_pixels = inputs["pixel_values"].to(
        model.device,
        dtype=torch.float16,
    )

    grid_thw = inputs["image_grid_thw"].to(
        model.device
    )

    print("sample:", sample_id)
    print("pixel_values shape:", tuple(clean_pixels.shape))
    print("grid_thw:", grid_thw.tolist())
    print(
        "clean pixel range:",
        float(clean_pixels.min()),
        float(clean_pixels.max()),
    )

    with torch.no_grad():
        clean_visual = model.visual(
            clean_pixels,
            grid_thw=grid_thw,
        ).detach()

    # Random start is required because MSE has zero gradient
    # when attacked_visual is initially identical to clean_visual.
    torch.manual_seed(42)

    random_delta = torch.empty_like(
        clean_pixels
    ).uniform_(
        -EPSILON,
        EPSILON,
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

        # Maximize token-wise representation difference.
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
                "No gradient was produced for pixel_values."
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

        print(
            "step =", step + 1,
            "representation_mse =",
            record["representation_mse"],
            "gradient_norm =",
            record["gradient_norm"],
            "max_abs_delta =",
            record["max_abs_delta"],
        )

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

    output = {
        "sample_id": sample_id,
        "model_id": MODEL_ID,
        "grid_thw": grid_thw.detach().cpu().tolist(),
        "pixel_values_shape": list(clean_pixels.shape),
        "visual_output_shape": list(clean_visual.shape),
        "steps": STEPS,
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "final_representation_mse": final_difference,
        "step_records": step_records,
    }

    output_path = (
        OUTPUT_ROOT
        / f"{sample_id}.json"
    )

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    # Save processed attack tensor only for smoke inspection.
    torch.save(
        {
            "sample_id": sample_id,
            "pixel_values": attacked.detach().cpu(),
            "image_grid_thw": grid_thw.detach().cpu(),
        },
        OUTPUT_ROOT / f"{sample_id}.pt",
    )

    initial_difference = step_records[0][
        "representation_mse"
    ]

    if not any(
        record["gradient_norm"] > 0
        for record in step_records
    ):
        raise RuntimeError(
            "Attack failed: all gradient norms are zero."
        )

    if final_difference <= initial_difference:
        raise RuntimeError(
            "Attack failed: representation difference "
            "did not increase."
        )

    print("\ninitial representation MSE:", initial_difference)
    print("final representation MSE:", final_difference)
    print("saved:", output_path)
    print("Qwen attack-gradient smoke test passed.")


if __name__ == "__main__":
    main()
