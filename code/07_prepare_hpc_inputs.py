'''
Epistemic code usually expects a simpler format:

folder_with_images/
question_file.jsonl

So this script prepares a clean HPC-ready version.
'''

from pathlib import Path
import json
import shutil

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_DATA_DIR = PROJECT_ROOT / "data" / "rohe_final"
FINAL_REGION_MAP_DIR = PROJECT_ROOT / "outputs" / "region_maps_final"

HPC_INPUT_DIR = PROJECT_ROOT / "outputs" / "hpc_inputs"

ORIGINAL_IMAGE_DIR = HPC_INPUT_DIR / "original_images"
REMOVED_IMAGE_DIR = HPC_INPUT_DIR / "removed_images"
REMOVED_IMAGE_JPG_DIR = HPC_INPUT_DIR / "removed_images_jpg"
TOKEN_REGION_DIR = HPC_INPUT_DIR / "token_regions"

QUESTIONS_ORIGINAL = HPC_INPUT_DIR / "questions_original.jsonl"
QUESTIONS_REMOVED = HPC_INPUT_DIR / "questions_removed.jsonl"
QUESTIONS_REMOVED_JPG = HPC_INPUT_DIR / "questions_removed_jpg.jsonl"

HPC_MANIFEST = HPC_INPUT_DIR / "hpc_manifest.json"


def reset_output_dir():
    if HPC_INPUT_DIR.exists():
        shutil.rmtree(HPC_INPUT_DIR)

    ORIGINAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REMOVED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REMOVED_IMAGE_JPG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_REGION_DIR.mkdir(parents=True, exist_ok=True)
    #original_images/      original COCO image, object present
    #removed_images/       LaMa image, object removed, still png
    #removed_images_jpg/   same removed image converted to jpg
    #token_regions/        token_to_region.json for each sample


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metadata_value(metadata: dict, keys: list[str], default=None):
    for key in keys:
        if key in metadata and metadata[key] not in [None, ""]:
            return metadata[key]
    return default

#decides the final question to ask
def normalize_question(question: str, target_object: str) -> str:
    if question:
        return question.strip()

    if target_object and target_object != "unknown":
        return f"Is there a {target_object} in the image?"

    return "Is there the target object in the image?"

#This converts removed image from PNG to JPG
def copy_removed_as_jpg(src_png: Path, dst_jpg: Path):
    img = Image.open(src_png).convert("RGB")
    img.save(dst_jpg, quality=95)


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    if not FINAL_DATA_DIR.exists():
        raise FileNotFoundError(f"Final data dir not found: {FINAL_DATA_DIR}")

    if not FINAL_REGION_MAP_DIR.exists():
        raise FileNotFoundError(f"Final region map dir not found: {FINAL_REGION_MAP_DIR}")

    reset_output_dir()

    sample_dirs = sorted(
        [p for p in FINAL_DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("sample_")]
    )

    original_rows = []
    removed_rows = []
    removed_jpg_rows = []
    manifest = []
    skipped = []

    print(f"Final samples found: {len(sample_dirs)}")
    print(f"HPC input dir: {HPC_INPUT_DIR}")
    print("-" * 60)

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name

        original_path = sample_dir / "original.jpg"
        removed_path = sample_dir / "removed.png"
        metadata_path = sample_dir / "metadata.json"

        region_dir = FINAL_REGION_MAP_DIR / sample_id
        token_region_path = region_dir / "token_to_region.json"
        region_counts_path = region_dir / "region_counts.json"

        required_paths = [
            original_path,
            removed_path,
            metadata_path,
            token_region_path,
            region_counts_path,
        ]

        missing = [p.name for p in required_paths if not p.exists()]

        if missing:
            skipped.append({
                "sample_id": sample_id,
                "reason": f"missing files: {missing}",
            })
            print(f"[SKIP] {sample_id}: missing {missing}")
            continue

        metadata = load_json(metadata_path)

        target_object = get_metadata_value(
            metadata,
            keys=[
                "target_object",
                "object",
                "category",
                "category_name",
                "target_class",
                "class_name",
            ],
            default="unknown",
        )

        question = get_metadata_value(
            metadata,
            keys=["question", "prompt"],
            default=None,
        )

        question = normalize_question(question, target_object)

        original_name = f"{sample_id}.jpg"
        removed_name = f"{sample_id}.png"
        removed_jpg_name = f"{sample_id}.jpg"

        dst_original = ORIGINAL_IMAGE_DIR / original_name
        dst_removed = REMOVED_IMAGE_DIR / removed_name
        dst_removed_jpg = REMOVED_IMAGE_JPG_DIR / removed_jpg_name

        shutil.copy2(original_path, dst_original)
        shutil.copy2(removed_path, dst_removed)
        copy_removed_as_jpg(removed_path, dst_removed_jpg)

        dst_token_region_dir = TOKEN_REGION_DIR / sample_id
        dst_token_region_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(token_region_path, dst_token_region_dir / "token_to_region.json")
        shutil.copy2(region_counts_path, dst_token_region_dir / "region_counts.json")
        shutil.copy2(metadata_path, dst_token_region_dir / "metadata.json")
        #Copy the key files needed later for region-wise masking

        original_row = {
            "question_id": f"{sample_id}_original",
            "sample_id": sample_id,
            "image": original_name,
            "image_file": original_name,
            "text": question,
            "question": question,
            "label": "yes",
            "answer": "yes",
            "target_object": target_object,
            "split": "original",
        }

        removed_row = {
            "question_id": f"{sample_id}_removed",
            "sample_id": sample_id,
            "image": removed_name,
            "image_file": removed_name,
            "text": question,
            "question": question,
            "label": "no",
            "answer": "no",
            "target_object": target_object,
            "split": "removed",
        }

        removed_jpg_row = {
            "question_id": f"{sample_id}_removed",
            "sample_id": sample_id,
            "image": removed_jpg_name,
            "image_file": removed_jpg_name,
            "text": question,
            "question": question,
            "label": "no",
            "answer": "no",
            "target_object": target_object,
            "split": "removed_jpg",
        }

        original_rows.append(original_row)
        removed_rows.append(removed_row)
        removed_jpg_rows.append(removed_jpg_row)

        manifest.append({
            "sample_id": sample_id,
            "target_object": target_object,
            "question": question,
            "original_image": str(dst_original),
            "removed_image": str(dst_removed),
            "removed_image_jpg": str(dst_removed_jpg),
            "token_region_dir": str(dst_token_region_dir),
        })

        print(f"[OK] prepared {sample_id}")

    write_jsonl(QUESTIONS_ORIGINAL, original_rows)
    write_jsonl(QUESTIONS_REMOVED, removed_rows)
    write_jsonl(QUESTIONS_REMOVED_JPG, removed_jpg_rows)

    with open(HPC_MANIFEST, "w", encoding="utf-8") as f:
        json.dump({
            "num_samples": len(manifest),
            "num_skipped": len(skipped),
            "samples": manifest,
            "skipped": skipped,
        }, f, indent=2)

    print("-" * 60)
    print("HPC inputs prepared.")
    print(f"Prepared samples: {len(manifest)}")
    print(f"Skipped samples:  {len(skipped)}")
    print(f"Original images:  {ORIGINAL_IMAGE_DIR}")
    print(f"Removed images:   {REMOVED_IMAGE_DIR}")
    print(f"Removed JPG:      {REMOVED_IMAGE_JPG_DIR}")
    print(f"Token regions:    {TOKEN_REGION_DIR}")
    print(f"Questions original:    {QUESTIONS_ORIGINAL}")
    print(f"Questions removed:     {QUESTIONS_REMOVED}")
    print(f"Questions removed JPG: {QUESTIONS_REMOVED_JPG}")
    print(f"HPC manifest:          {HPC_MANIFEST}")


if __name__ == "__main__":
    main()