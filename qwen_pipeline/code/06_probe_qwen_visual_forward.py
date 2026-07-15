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
    "hpc_inputs_smoke/questions_removed_jpg.jsonl"
)

IMAGE_ROOT = Path(
    "hpc_inputs_smoke/removed_images_jpg"
)


def read_first_row(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(next(handle))


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

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    row = read_first_row(QUESTIONS)
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
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(
        model.device,
        dtype=torch.float16,
    )

    image_grid_thw = inputs["image_grid_thw"].to(
        model.device
    )

    print("pixel_values shape:", tuple(pixel_values.shape))
    print("image_grid_thw:", image_grid_thw.tolist())

    with torch.inference_mode():
        visual_output = model.visual(
            pixel_values,
            grid_thw=image_grid_thw,
        )

    if isinstance(visual_output, tuple):
        print("visual output type: tuple")
        for index, value in enumerate(visual_output):
            if torch.is_tensor(value):
                print(
                    f"tuple[{index}] shape:",
                    tuple(value.shape),
                )
            else:
                print(
                    f"tuple[{index}] type:",
                    type(value),
                )
    elif torch.is_tensor(visual_output):
        print(
            "visual output shape:",
            tuple(visual_output.shape),
        )
    else:
        print(
            "visual output type:",
            type(visual_output),
        )

    expected_merged = (
        int(image_grid_thw[0, 0])
        * int(image_grid_thw[0, 1])
        * int(image_grid_thw[0, 2])
        // 4
    )

    print("expected merged tokens:", expected_merged)
    print("Qwen independent visual forward passed.")


if __name__ == "__main__":
    main()
