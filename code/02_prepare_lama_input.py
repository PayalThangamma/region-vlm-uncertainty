'''
This script basically puts the input imaged from data set to perform inpainting on to get the image with target object inpainted
using lama before runnung lama on it.
'''

from pathlib import Path
import shutil

DATA = Path("../data/rohe_raw")
OUT = Path("../lama_input")


def clear_output_folder():
    OUT.mkdir(parents=True, exist_ok=True)

    for file_path in OUT.glob("*"):
        if file_path.is_file():
            file_path.unlink()


def prepare_sample(sample_dir):
    sample_id = sample_dir.name

    original = sample_dir / "original.jpg"
    mask = sample_dir / "mask.png"

    if not original.exists() or not mask.exists():
        print(f"Skipping {sample_id}: missing original.jpg or mask.png")
        return False

    shutil.copy(original, OUT / f"{sample_id}.png")
    shutil.copy(mask, OUT / f"{sample_id}_mask.png")

    return True


def main():
    clear_output_folder()
    count = 0
    for sample_dir in sorted(DATA.glob("sample_*")):
        if prepare_sample(sample_dir):
            count += 1
    print(f"Prepared {count} samples for LaMa in {OUT}")


if __name__ == "__main__":
    main()