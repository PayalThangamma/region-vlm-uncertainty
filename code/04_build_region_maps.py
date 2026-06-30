from pathlib import Path
import json
import shutil

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#parents[1] means go one level above code/
RAW_DATA_DIR = PROJECT_ROOT / "data" / "rohe_raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "region_maps_rohe"

IMAGE_SIZE = 336
#LLaVA-1.5 uses CLIP vision input size 336×336 therefore all masks must be resized to 336×336
PATCH_SIZE = 14
#CLIP ViT-L/14 uses patch size 14 meaning one visual token corresponds to one 14×14 image patch
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
# 336/14=24 meaning visual token grid is 24 * 24
EXPECTED_TOKENS = GRID_SIZE * GRID_SIZE
# each image has 24 * 24 = 576 patch tokens.
DILATION_SIZE = 35
OVERLAP_THRESHOLD = 0.25
#one patch = 14 pixels meaning 35 pixels is 2.5 patches
#token = removed () if 30% of patch inside removed object) so that it is not removed region when 1 tiny pixel touches the object


def load_removed_mask(mask_path: Path) -> np.ndarray:
    """
    Load COCO object mask and resize it to 336x336.

    This is important because LLaVA CLIP-L/336 uses:
    336 / 14 = 24 patches per side
    24 * 24 = 576 patch tokens
    """
    img = Image.open(mask_path).convert("L") #Open the mask image and convert it to grayscale
    try:
        resample_mode = Image.Resampling.NEAREST
    except AttributeError:
        resample_mode = Image.NEAREST
    #nearest-neighbor resizing we do not want blurred edges in the mask
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=resample_mode)
    mask = np.array(img) > 127 #Converts image to a boolean mask pixels above 127 become Ture
    return mask


def create_context_mask(removed_mask: np.ndarray, dilation_size: int = DILATION_SIZE) -> np.ndarray:
    """
    Context region = area around the removed object.

    We dilate the removed object mask, then subtract the original removed mask to get context ring.
    """
    if dilation_size % 2 == 0:
        dilation_size += 1
    #ImageFilter.MaxFilter needs an odd filter size, if someone gives 34, we convert it to 35
    removed_img = Image.fromarray((removed_mask.astype(np.uint8) * 255))
    #Convert boolean mask back into an image
    dilated_img = removed_img.filter(ImageFilter.MaxFilter(dilation_size))
    #dilates the removed object mask (object + surrounding region)
    dilated = np.array(dilated_img) > 127
    #dilate dimage back to boolean
    context = dilated & (~removed_mask)
    #context = dilated object area minus original object area
    return context


def create_background_mask(removed_mask: np.ndarray, context_mask: np.ndarray) -> np.ndarray:
    """
    Background = everything that is not removed-object region and not context region.
    """
    return ~(removed_mask | context_mask)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    img.save(path)


def map_tokens_to_regions(
    removed_mask: np.ndarray,
    context_mask: np.ndarray,
    background_mask: np.ndarray,
    patch_size: int = PATCH_SIZE,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> dict:
    """
    Map each visual patch token to one semantic region.

    Token order:
    token 0 = top-left patch
    token 575 = bottom-right patch

    Note:
    This maps only 576 patch tokens.
    It does not include the CLS token.
    """
    token_to_region = {}

    token_id = 0

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y1 = row * patch_size
            y2 = y1 + patch_size
            x1 = col * patch_size
            x2 = x1 + patch_size
            #pixel coordinates of the current token patch
            removed_patch = removed_mask[y1:y2, x1:x2]
            context_patch = context_mask[y1:y2, x1:x2]
            background_patch = background_mask[y1:y2, x1:x2]
            #Extracts the same 14×14 patch from each region mask
            patch_area = patch_size * patch_size
            #one patch area = 14 × 14 = 196 pixels
            removed_overlap = removed_patch.sum() / patch_area
            context_overlap = context_patch.sum() / patch_area
            background_overlap = background_patch.sum() / patch_area
            #Calculate how much of the patch belongs to each region.
            if removed_overlap >= overlap_threshold:
                region = "removed"
            elif context_overlap >= overlap_threshold:
                region = "context"
            else:
                region = "background"
            #Why removed first-Because if a token touches the removed object enough, we want to call it a removed-object token.
            # Why context second-Because context is the next most important region.
            token_to_region[str(token_id)] = region
            token_id += 1

    return token_to_region


def count_regions(token_to_region: dict) -> dict:
    counts = {
        "removed": 0,
        "context": 0,
        "background": 0,
    }

    for region in token_to_region.values():
        counts[region] += 1

    counts["total"] = sum(counts.values())
    #shoudl always by 576
    return counts

def main():
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(
        [p for p in RAW_DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("sample_")]
    )

    manifest = []
    skipped = []

    print(f"Raw data dir: {RAW_DATA_DIR}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print(f"Samples found: {len(sample_dirs)}")
    print("-" * 60)

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name

        original_path = sample_dir / "original.jpg"
        removed_path = sample_dir / "removed.png"
        mask_path = sample_dir / "mask.png"
        metadata_path = sample_dir / "metadata.json"

        missing = []
        for path in [original_path, removed_path, mask_path, metadata_path]:
            if not path.exists():
                missing.append(path.name)

        if missing:
            skipped.append({
                "sample_id": sample_id,
                "reason": f"missing files: {missing}",
            })
            print(f"[SKIP] {sample_id}: missing {missing}")
            continue

        removed_mask = load_removed_mask(mask_path)

        if removed_mask.sum() == 0:
            skipped.append({
                "sample_id": sample_id,
                "reason": "empty removed mask after resize",
            })
            print(f"[SKIP] {sample_id}: empty removed mask")
            continue

        context_mask = create_context_mask(removed_mask)
        background_mask = create_background_mask(removed_mask, context_mask)

        token_to_region = map_tokens_to_regions(
            removed_mask=removed_mask,
            context_mask=context_mask,
            background_mask=background_mask,
        )

        if len(token_to_region) != EXPECTED_TOKENS:
            skipped.append({
                "sample_id": sample_id,
                "reason": f"wrong token count: {len(token_to_region)}",
            })
            print(f"[SKIP] {sample_id}: wrong token count")
            continue

        region_counts = count_regions(token_to_region)

        if region_counts["total"] != EXPECTED_TOKENS:
            skipped.append({
                "sample_id": sample_id,
                "reason": f"region count does not sum to {EXPECTED_TOKENS}",
            })
            print(f"[SKIP] {sample_id}: bad region count")
            continue

        out_sample_dir = OUTPUT_DIR / sample_id
        out_sample_dir.mkdir(parents=True, exist_ok=True)

        save_binary_mask(removed_mask, out_sample_dir / "removed_mask.png")
        save_binary_mask(context_mask, out_sample_dir / "context_mask.png")
        save_binary_mask(background_mask, out_sample_dir / "background_mask.png")

        with open(out_sample_dir / "token_to_region.json", "w", encoding="utf-8") as f:
            json.dump(token_to_region, f, indent=2)

        with open(out_sample_dir / "region_counts.json", "w", encoding="utf-8") as f:
            json.dump(region_counts, f, indent=2)

        shutil.copy2(metadata_path, out_sample_dir / "metadata.json")
        #Copy the sample metadata. This keeps question/object information attached to region maps.
        manifest.append({
            "sample_id": sample_id,
            "region_map_dir": str(out_sample_dir),
            "removed_tokens": region_counts["removed"],
            "context_tokens": region_counts["context"],
            "background_tokens": region_counts["background"],
            "total_tokens": region_counts["total"],
        })

        print(
            f"[OK] {sample_id}: "
            f"removed={region_counts['removed']}, "
            f"context={region_counts['context']}, "
            f"background={region_counts['background']}, "
            f"total={region_counts['total']}"
        )

    manifest_path = OUTPUT_DIR / "manifest_region_maps.json"
    skipped_path = OUTPUT_DIR / "skipped_region_maps.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(skipped_path, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)

    print("-" * 60)
    print(f"Done.")
    print(f"Successful samples: {len(manifest)}")
    print(f"Skipped samples:    {len(skipped)}")
    print(f"Manifest saved to:  {manifest_path}")
    print(f"Skipped saved to:   {skipped_path}")


if __name__ == "__main__":
    main()