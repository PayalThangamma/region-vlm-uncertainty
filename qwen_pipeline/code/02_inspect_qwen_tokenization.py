import json
from pathlib import Path

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

QUESTIONS = Path(
    "hpc_inputs_smoke/questions_removed_jpg.jsonl"
)
IMAGE_ROOT = Path(
    "hpc_inputs_smoke/removed_images_jpg"
)


def read_rows(path: Path, limit: int = 3) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

            if len(rows) == limit:
                break

    return rows


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    image_processor = processor.image_processor

    print("Processor class:", type(processor))
    print("Image processor class:", type(image_processor))

    for attribute in [
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "min_pixels",
        "max_pixels",
    ]:
        print(
            attribute,
            "=",
            getattr(image_processor, attribute, None),
        )

    print(
        "image_token_id =",
        getattr(processor.tokenizer, "image_token_id", None),
    )

    image_token = "<|image_pad|>"
    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        image_token
    )

    print("<|image_pad|> token ID =", image_token_id)

    for row in read_rows(QUESTIONS):
        sample_id = row["sample_id"]
        image_path = IMAGE_ROOT / row["image"]
        question = row.get("text", row.get("prompt"))

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

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        grid = inputs["image_grid_thw"][0].tolist()
        temporal, height, width = grid

        premerge_tokens = temporal * height * width
        merge_size = getattr(
            image_processor,
            "merge_size",
            2,
        )
        merged_tokens = (
            premerge_tokens // (merge_size ** 2)
        )

        actual_image_tokens = int(
            (inputs["input_ids"] == image_token_id)
            .sum()
            .item()
        )

        print("\nSample:", sample_id)
        print("image:", image_path)
        print("grid_thw:", grid)
        print("premerge tokens:", premerge_tokens)
        print("expected merged tokens:", merged_tokens)
        print("actual image-pad tokens:", actual_image_tokens)
        print(
            "input sequence length:",
            inputs["input_ids"].shape[1],
        )

        if actual_image_tokens != merged_tokens:
            raise ValueError(
                f"{sample_id}: expected {merged_tokens} "
                f"image tokens but found {actual_image_tokens}"
            )

    print("\nQwen visual-token counts verified.")


if __name__ == "__main__":
    main()
