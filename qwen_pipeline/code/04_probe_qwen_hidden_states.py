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

ATTACK_ROOT = Path(
    "qwen_pipeline/outputs/aligned_attack_geometry_smoke"
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


class ForwardHook:
    def __init__(self, module):
        self.output = None
        self.handle = module.register_forward_hook(
            self._hook
        )

    def _hook(self, module, inputs, output):
        self.output = output

    def close(self):
        self.handle.remove()


def get_hidden_tensor(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, tuple):
        for item in output:
            if torch.is_tensor(item):
                return item

    raise TypeError(
        f"Could not extract tensor from output type {type(output)}"
    )


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

    if not hasattr(model, "visual"):
        raise AttributeError(
            "Expected Qwen model to have model.visual"
        )

    if not hasattr(model.visual, "blocks"):
        raise AttributeError(
            "Expected model.visual.blocks"
        )

    num_blocks = len(model.visual.blocks)
    print("Number of visual blocks:", num_blocks)

    layer_indices = [
        0,
        num_blocks // 4,
        num_blocks // 2,
        3 * num_blocks // 4,
        num_blocks - 1,
    ]

    layer_indices = sorted(set(layer_indices))
    print("Hooked layers:", layer_indices)

    hooks = {
        index: ForwardHook(
            model.visual.blocks[index]
        )
        for index in layer_indices
    }

    rows = read_rows(QUESTIONS)

    for row in rows:
        sample_id = row["sample_id"]
        question = row.get(
            "text",
            row.get("prompt"),
        )

        clean_path = IMAGE_ROOT / row["image"]

        attack_path = (
            ATTACK_ROOT
            / f"{Path(row['image']).stem}.png"
        )

        if not clean_path.exists():
            raise FileNotFoundError(clean_path)

        if not attack_path.exists():
            raise FileNotFoundError(attack_path)

        clean_inputs = build_inputs(
            processor,
            clean_path,
            question,
        )

        attack_inputs = build_inputs(
            processor,
            attack_path,
            question,
        )

        clean_grid = clean_inputs[
            "image_grid_thw"
        ][0].tolist()

        attack_grid = attack_inputs[
            "image_grid_thw"
        ][0].tolist()

        print("\nSample:", sample_id)
        print("clean grid:", clean_grid)
        print("attack grid:", attack_grid)

        if clean_grid != attack_grid:
            raise ValueError(
                f"{sample_id}: clean and attack grids differ"
            )

        clean_inputs = {
            key: value.to(model.device)
            if hasattr(value, "to")
            else value
            for key, value in clean_inputs.items()
        }

        attack_inputs = {
            key: value.to(model.device)
            if hasattr(value, "to")
            else value
            for key, value in attack_inputs.items()
        }

        with torch.inference_mode():
            _ = model(
                **clean_inputs,
                use_cache=False,
                return_dict=True,
            )

        clean_hidden = {
            index: get_hidden_tensor(
                hooks[index].output
            ).detach().float().cpu()
            for index in layer_indices
        }

        with torch.inference_mode():
            _ = model(
                **attack_inputs,
                use_cache=False,
                return_dict=True,
            )

        attack_hidden = {
            index: get_hidden_tensor(
                hooks[index].output
            ).detach().float().cpu()
            for index in layer_indices
        }

        merged_token_count = (
            clean_grid[0]
            * clean_grid[1]
            * clean_grid[2]
            // (
                processor.image_processor.merge_size
                ** 2
            )
        )

        print(
            "expected merged tokens:",
            merged_token_count,
        )

        for index in layer_indices:
            clean_tensor = clean_hidden[index]
            attack_tensor = attack_hidden[index]

            print(
                "layer",
                index,
                "clean shape =",
                tuple(clean_tensor.shape),
                "attack shape =",
                tuple(attack_tensor.shape),
            )

            if clean_tensor.shape != attack_tensor.shape:
                raise ValueError(
                    f"{sample_id}, layer {index}: "
                    "clean/attack shape mismatch"
                )

    for hook in hooks.values():
        hook.close()

    print("\nQwen visual hidden-state probe passed.")


if __name__ == "__main__":
    main()
