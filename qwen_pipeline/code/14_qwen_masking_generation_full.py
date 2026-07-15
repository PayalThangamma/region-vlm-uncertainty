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

UNCERTAINTY_ROOT = Path(
    "qwen_pipeline/outputs/uncertainty_removed_full"
)

REGION_ROOT = Path(
    "qwen_pipeline/outputs/token_regions_full"
)

OUTPUT_ROOT = Path(
    "qwen_pipeline/outputs/eval_removed_full"
)

CONDITIONS = [
    "none",
    "all",
    "removed",
    "context",
    "background",
]

NUM_SAMPLES = None
MAX_NEW_TOKENS = 32


def read_rows(path: Path, limit=None) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

            if limit is not None and len(rows) >= limit:
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
        padding=True,
        return_tensors="pt",
    )


class VisualMaskHook:
    def __init__(self, module):
        self.suppress_mask = None
        self.handle = module.register_forward_hook(
            self._hook
        )

    def _hook(self, module, inputs, output):
        if self.suppress_mask is None:
            return output

        if not torch.is_tensor(output):
            raise TypeError(
                f"Expected tensor visual output, found {type(output)}"
            )

        if output.ndim != 2:
            raise ValueError(
                f"Expected 2D visual output, found {output.shape}"
            )

        if output.shape[0] != self.suppress_mask.numel():
            raise ValueError(
                "Visual-token mismatch: "
                f"output has {output.shape[0]} tokens, "
                f"mask has {self.suppress_mask.numel()}"
            )

        keep = (~self.suppress_mask).to(
            device=output.device,
            dtype=output.dtype,
        )

        return output * keep.unsqueeze(-1)

    def close(self):
        self.handle.remove()


def build_suppress_mask(
    condition: str,
    uncertainty_data: dict,
    token_to_region: dict,
) -> torch.Tensor:
    scores = torch.tensor(
        uncertainty_data["merged_uncertainty"],
        dtype=torch.float32,
    )

    threshold = float(
        uncertainty_data["threshold"]
    )

    uncertain = scores >= threshold

    if condition == "none":
        return torch.zeros_like(
            uncertain,
            dtype=torch.bool,
        )

    if condition == "all":
        return uncertain

    region_mask = torch.zeros_like(
        uncertain,
        dtype=torch.bool,
    )

    for token_id, region in token_to_region.items():
        token_index = int(token_id)

        if region == condition:
            region_mask[token_index] = True

    return uncertain & region_mask


def classify_answer(text: str) -> str:
    normalized = text.strip().lower()

    if normalized.startswith("yes"):
        return "yes"

    if normalized.startswith("no"):
        return "no"

    return "unknown"


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
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    hook = VisualMaskHook(model.visual)

    rows = read_rows(
        QUESTIONS,
        NUM_SAMPLES,
    )

    print("Samples to process:", len(rows))

    outputs = {
        condition: (
            OUTPUT_ROOT
            / f"captions_{condition}.jsonl"
        ).open("w", encoding="utf-8")
        for condition in CONDITIONS
    }

    try:
        for row in rows:
            sample_id = row["sample_id"]
            image_path = IMAGE_ROOT / row["image"]
            question = row.get(
                "text",
                row.get("prompt"),
            )

            uncertainty_path = (
                UNCERTAINTY_ROOT
                / f"{sample_id}.json"
            )

            region_path = (
                REGION_ROOT
                / sample_id
                / "token_to_region.json"
            )

            uncertainty_data = json.loads(
                uncertainty_path.read_text(
                    encoding="utf-8"
                )
            )

            token_to_region = json.loads(
                region_path.read_text(
                    encoding="utf-8"
                )
            )

            inputs = build_inputs(
                processor,
                image_path,
                question,
            )

            actual_grid = inputs[
                "image_grid_thw"
            ][0].tolist()

            expected_grid = uncertainty_data[
                "image_grid_thw"
            ]

            if actual_grid != expected_grid:
                raise ValueError(
                    f"{sample_id}: grid mismatch: "
                    f"{actual_grid} vs {expected_grid}"
                )

            inputs = {
                key: value.to(model.device)
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

            for condition in CONDITIONS:
                suppress_mask = build_suppress_mask(
                    condition,
                    uncertainty_data,
                    token_to_region,
                )

                hook.suppress_mask = suppress_mask

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )

                generated_only = generated_ids[
                    :,
                    inputs["input_ids"].shape[1]:,
                ]

                answer = processor.batch_decode(
                    generated_only,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0].strip()

                result = {
                    "question_id":
                        row.get("question_id"),
                    "sample_id": sample_id,
                    "image": row["image"],
                    "prompt": question,
                    "text": answer,
                    "answer_class":
                        classify_answer(answer),
                    "model_id": MODEL_ID,
                    "label": row.get("label"),
                    "target_object":
                        row.get("target_object"),
                    "split": row.get("split"),
                    "condition": condition,
                    "num_visual_tokens":
                        int(suppress_mask.numel()),
                    "num_suppressed_tokens":
                        int(
                            suppress_mask.sum().item()
                        ),
                }

                outputs[condition].write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                outputs[condition].flush()

                print(
                    sample_id,
                    "condition =",
                    condition,
                    "suppressed =",
                    result[
                        "num_suppressed_tokens"
                    ],
                    "answer =",
                    answer,
                )

    finally:
        hook.suppress_mask = None
        hook.close()

        for handle in outputs.values():
            handle.close()

    print(
        "\nQwen five-condition masking smoke test passed."
    )


if __name__ == "__main__":
    main()
