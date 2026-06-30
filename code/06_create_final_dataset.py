from pathlib import Path
import csv
import json
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "rohe_raw"
RAW_REGION_MAP_DIR = PROJECT_ROOT / "outputs" / "region_maps_rohe"
QUALITY_CSV = PROJECT_ROOT / "outputs" / "rohe_quality.csv"
#quality csv example row from script 5: sample_000107,131,55,390,576,True,

FINAL_DATA_DIR = PROJECT_ROOT / "data" / "rohe_final"
FINAL_REGION_MAP_DIR = PROJECT_ROOT / "outputs" / "region_maps_final"

FINAL_MANIFEST = PROJECT_ROOT / "outputs" / "final_dataset_manifest.json"


def read_good_samples():
    if not QUALITY_CSV.exists():
        raise FileNotFoundError(f"Quality CSV not found: {QUALITY_CSV}")

    good_samples = []

    with open(QUALITY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            good_value = str(row["good_sample"]).strip().lower()

            if good_value in ["true", "1", "yes"]:
                good_samples.append(row["sample_id"])

    return good_samples

#prepares clean final output folders
def reset_output_dirs():
    if FINAL_DATA_DIR.exists():
        shutil.rmtree(FINAL_DATA_DIR)

    if FINAL_REGION_MAP_DIR.exists():
        shutil.rmtree(FINAL_REGION_MAP_DIR)

    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_REGION_MAP_DIR.mkdir(parents=True, exist_ok=True)


def copy_sample(sample_id: str):
    src_sample_dir = RAW_DATA_DIR / sample_id
    src_region_dir = RAW_REGION_MAP_DIR / sample_id

    dst_sample_dir = FINAL_DATA_DIR / sample_id
    dst_region_dir = FINAL_REGION_MAP_DIR / sample_id

    if not src_sample_dir.exists():
        raise FileNotFoundError(f"Missing raw sample directory: {src_sample_dir}")

    if not src_region_dir.exists():
        raise FileNotFoundError(f"Missing region map directory: {src_region_dir}")

    shutil.copytree(src_sample_dir, dst_sample_dir)
    shutil.copytree(src_region_dir, dst_region_dir)
    #Used for copying and deleting folders


def main():
    good_samples = read_good_samples()

    print(f"Good samples found in quality CSV: {len(good_samples)}")

    reset_output_dirs()

    manifest = []

    for sample_id in good_samples:
        copy_sample(sample_id)

        manifest.append({
            "sample_id": sample_id,
            "sample_dir": str(FINAL_DATA_DIR / sample_id),
            "region_map_dir": str(FINAL_REGION_MAP_DIR / sample_id),
        })

        print(f"[OK] copied {sample_id}")

    with open(FINAL_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("-" * 60)
    print("Final common dataset created.")
    print(f"Final samples: {len(manifest)}")
    print(f"Final data dir: {FINAL_DATA_DIR}")
    print(f"Final region map dir: {FINAL_REGION_MAP_DIR}")
    print(f"Final manifest: {FINAL_MANIFEST}")


if __name__ == "__main__":
    main()