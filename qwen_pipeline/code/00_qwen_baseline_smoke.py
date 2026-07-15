import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def read_jsonl(path: Path, limit: int) -> list[dict]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    print("Loading model:", MODEL_ID)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    questions = read_jsonl(args.questions, args.num_samples)

    if len(questions) != args.num_samples:
        raise ValueError(
            f"Expected {args.num_samples} samples, found {len(questions)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        for row in questions:
            sample_id = row["sample_id"]
            image_name = row["image"]
            question = row.get("text", row.get("prompt"))

            if question is None:
                raise KeyError(f"No question field for {sample_id}")

            image_path = args.image_root / image_name

            if not image_path.exists():
                raise FileNotFoundError(image_path)

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

            inputs = {
                key: value.to(model.device)
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

            print("\nSample:", sample_id)
            print("input_ids shape:", tuple(inputs["input_ids"].shape))

            if "image_grid_thw" in inputs:
                print(
                    "image_grid_thw:",
                    inputs["image_grid_thw"].detach().cpu().tolist(),
                )

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
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
                "question_id": row.get("question_id"),
                "sample_id": sample_id,
                "image": image_name,
                "prompt": question,
                "text": answer,
                "model_id": MODEL_ID,
                "label": row.get("label"),
                "target_object": row.get("target_object"),
                "split": row.get("split"),
            }

            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            print("Question:", question)
            print("Answer:", answer)

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
