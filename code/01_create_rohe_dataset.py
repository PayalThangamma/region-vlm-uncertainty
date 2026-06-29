
'''
This script basically creates filtered samples that we want to create our pipeline from COCO images using the ROHE idea of
masking the target object and evaluating on POPE
'''

from pathlib import Path
import json
import shutil

import numpy as np
from PIL import Image
from pycocotools.coco import COCO #to read COCO annotations and convert segmentation annotations into masks


COCO_IMG_DIR = Path("../data/coco/val2017")
COCO_ANN = Path("../data/coco/annotations/instances_val2017.json")
OUT_ROOT = Path("../data/rohe_raw")
#i/o paths


TARGET_CLASSES = ["dog", "cat", "car", "chair", "bicycle", "bus", "bottle", "bench"]
#object classes chosen because these are common COCO classes with clear visual objects.
MAX_CANDIDATES = 636

MIN_OBJECT_FRAC = 0.05
MAX_OBJECT_FRAC = 0.30


def load_coco():
    return COCO(str(COCO_ANN))


def collect_candidates(coco):
    candidates = []
    cat_ids = coco.getCatIds(catNms=TARGET_CLASSES)
    #loads COCO annotations and category IDs for target classes

    for cat_id in cat_ids:
        category = coco.loadCats(cat_id)[0]["name"]
        img_ids = coco.getImgIds(catIds=[cat_id])
        #gets image ids for each category

        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            image_area = img_info["width"] * img_info["height"]
            #loads relevant image

            ann_ids = coco.getAnnIds(
                imgIds=[img_id],
                catIds=[cat_id],
                iscrowd=False,
            )
            anns = coco.loadAnns(ann_ids)
            #gets the annoation ids that are non messy in the image

            if not anns:
                continue 
            #skip incase of absent annotations

            largest_ann = max(anns, key=lambda a: a["area"])
            object_fraction = largest_ann["area"] / image_area
            #choose the largest becuase small instances may be too tiny for token mapping and compute occupancy in image 

            if MIN_OBJECT_FRAC <= object_fraction <= MAX_OBJECT_FRAC:
                candidates.append({
                    "image_id": img_id,
                    "file_name": img_info["file_name"],
                    "target_object": category,
                    "ann_id": largest_ann["id"],
                    "area": largest_ann["area"],
                    "object_fraction": object_fraction,
                })
            #choose medium sized objects

    return sorted(
        candidates,
        key=lambda x: x["object_fraction"],
        reverse=True,
    )[:MAX_CANDIDATES] #sorting in descending size of objects


def create_overlay(original_path, mask): #creates red overlay on mask
    original = Image.open(original_path).convert("RGB")
    original_np = np.array(original)
    overlay = original_np.copy()

    overlay[mask > 0] = (
        0.6 * overlay[mask > 0] + 0.4 * np.array([255, 0, 0])
    ).astype(np.uint8)

    return Image.fromarray(overlay)


def create_sample(coco, sample, sample_id):
    sample_dir = OUT_ROOT / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    src_img = COCO_IMG_DIR / sample["file_name"]
    dst_img = sample_dir / "original.jpg"

    if not src_img.exists():
        print(f"Skipping {sample_id}: missing image {src_img}")
        return None

    shutil.copy(src_img, dst_img)

    ann = coco.loadAnns([sample["ann_id"]])[0]
    mask = coco.annToMask(ann).astype(np.uint8) * 255

    Image.fromarray(mask).save(sample_dir / "mask.png")

    overlay = create_overlay(dst_img, mask)
    overlay.save(sample_dir / "mask_overlay.png")

    #stores metadata for later help
    metadata = {
        "sample_id": sample_id,
        "original_image": "original.jpg",
        "removed_image": "removed.png",
        "mask_image": "mask.png",
        "source_dataset": "COCO val2017",
        "source_image_id": sample["image_id"],
        "source_file_name": sample["file_name"],
        "target_object": sample["target_object"],
        "question": f"Is there a {sample['target_object']} in the image?",
        "ground_truth": "no",
        "sample_type": "object_removal",
        "coco_annotation_id": sample["ann_id"],
        "coco_area": sample["area"],
        "object_fraction": sample["object_fraction"],
    }

    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def save_manifest(manifest):
    with open(OUT_ROOT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    coco = load_coco()
    candidates = collect_candidates(coco)

    manifest = []

    for idx, sample in enumerate(candidates, start=1):
        sample_id = f"sample_{idx:06d}"
        metadata = create_sample(coco, sample, sample_id)

        if metadata is None:
            continue

        manifest.append(metadata)

        print(
            sample_id,
            sample["target_object"],
            sample["file_name"],
            "fraction:",
            round(sample["object_fraction"], 3),
        )

    save_manifest(manifest)
    print("Created", len(manifest), "samples in", OUT_ROOT)


if __name__ == "__main__":
    main()