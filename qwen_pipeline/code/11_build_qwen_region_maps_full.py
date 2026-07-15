import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

QUESTIONS_PATH = Path(
    "hpc_inputs/questions_removed_jpg.jsonl"
)

IMAGE_ROOT = Path(
    "hpc_inputs/removed_images_jpg"
)

# Existing semantic masks from the LLaVA pipeline.
REGION_ROOT = Path(
    "outputs/region_maps_final"
)

OUTPUT_ROOT = Path(
    "qwen_pipeline/outputs/token_regions_full"
)

REGIONS = [
    "removed",
    "context",
    "background",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    image = Image.open(path).convert("L")
    array = np.asarray(image)

    return array > 127


def resize_binary_mask(
    mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    image = Image.fromarray(
        (mask.astype(np.uint8) * 255)
    )

    resized = image.resize(
        (width, height),
        resample=Image.Resampling.NEAREST,
    )

    return np.asarray(resized) > 127


def assign_premerge_regions(
    removed: np.ndarray,
    context: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    height, width = removed.shape

    output = np.empty(
        (height, width),
        dtype=object,
    )

    # Priority prevents overlaps from creating ambiguous labels.
    output[:, :] = "background"
    output[context] = "context"
    output[removed] = "removed"

    if not np.all(
        removed | context | background
    ):
        raise ValueError(
            "Some resized patch positions have no semantic region."
        )

    return output


def merge_regions(
    region_grid: np.ndarray,
    merge_size: int,
) -> list[str]:
    height, width = region_grid.shape

    if height % merge_size != 0:
        raise ValueError(
            f"Height {height} is not divisible by {merge_size}"
        )

    if width % merge_size != 0:
        raise ValueError(
            f"Width {width} is not divisible by {merge_size}"
        )

    merged_labels = []

    for row in range(0, height, merge_size):
        for column in range(0, width, merge_size):
            block = region_grid[
                row:row + merge_size,
                column:column + merge_size,
            ].reshape(-1)

            counts = {
                region: int(
                    np.sum(block == region)
                )
                for region in REGIONS
            }

            # Deterministic priority for ties:
            # removed > context > background.
            label = max(
                REGIONS,
                key=lambda region: (
                    counts[region],
                    -REGIONS.index(region),
                ),
            )

            merged_labels.append(label)

    return merged_labels


def main() -> None:
    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    merge_size = int(
        processor.image_processor.merge_size
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_jsonl(QUESTIONS_PATH)

    for row in rows:
        sample_id = row["sample_id"]
        image_path = IMAGE_ROOT / row["image"]

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
                        "text": row.get(
                            "text",
                            row.get("prompt"),
                        ),
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

        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        temporal, grid_h, grid_w = (
            inputs["image_grid_thw"][0]
            .tolist()
        )

        if temporal != 1:
            raise ValueError(
                f"{sample_id}: expected temporal grid 1, "
                f"found {temporal}"
            )

        region_folder = REGION_ROOT / sample_id

        removed = load_mask(
            region_folder / "removed_mask.png"
        )
        context = load_mask(
            region_folder / "context_mask.png"
        )
        background = load_mask(
            region_folder / "background_mask.png"
        )

        removed_grid = resize_binary_mask(
            removed,
            width=grid_w,
            height=grid_h,
        )

        context_grid = resize_binary_mask(
            context,
            width=grid_w,
            height=grid_h,
        )

        background_grid = resize_binary_mask(
            background,
            width=grid_w,
            height=grid_h,
        )

        premerge_regions = assign_premerge_regions(
            removed_grid,
            context_grid,
            background_grid,
        )

        merged_labels = merge_regions(
            premerge_regions,
            merge_size=merge_size,
        )

        expected_tokens = (
            temporal
            * grid_h
            * grid_w
            // (merge_size ** 2)
        )

        if len(merged_labels) != expected_tokens:
            raise ValueError(
                f"{sample_id}: expected "
                f"{expected_tokens} merged labels, "
                f"found {len(merged_labels)}"
            )

        token_to_region = {
            str(index): region
            for index, region in enumerate(
                merged_labels
            )
        }

        counts = {
            region: merged_labels.count(region)
            for region in REGIONS
        }

        sample_output = OUTPUT_ROOT / sample_id
        sample_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            sample_output
            / "token_to_region.json"
        ).open("w", encoding="utf-8") as handle:
            json.dump(
                token_to_region,
                handle,
                indent=2,
            )

        metadata = {
            "sample_id": sample_id,
            "image": row["image"],
            "image_grid_thw": [
                temporal,
                grid_h,
                grid_w,
            ],
            "merge_size": merge_size,
            "premerge_token_count":
                temporal * grid_h * grid_w,
            "merged_token_count":
                len(merged_labels),
            "region_counts": counts,
        }

        with (
            sample_output
            / "metadata.json"
        ).open("w", encoding="utf-8") as handle:
            json.dump(
                metadata,
                handle,
                indent=2,
            )

        print(
            sample_id,
            "grid =",
            [temporal, grid_h, grid_w],
            "merged =",
            len(merged_labels),
            "counts =",
            counts,
        )

    print(
        "\nQwen smoke region maps created:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
