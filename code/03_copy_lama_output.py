'''
This script uses the lama output images after running lama on the lama input images to copy the lama output images back as 
removed.png to the original sample folders in the dataset
'''

from pathlib import Path
import shutil

DATA = Path("../data/rohe_raw")
LAMA_OUT = Path("../lama_output")


def find_lama_output(sample_id):
    possible_paths = [
        LAMA_OUT / f"{sample_id}.png",
        LAMA_OUT / f"{sample_id}.jpg",
        LAMA_OUT / f"{sample_id}_mask.png",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def copy_output_to_sample(sample_dir):
    sample_id = sample_dir.name
    output_path = find_lama_output(sample_id)

    if output_path is None:
        print(f"Missing LaMa output for {sample_id}")
        return False

    shutil.copy(output_path, sample_dir / "removed.png")
    return True


def main():
    count = 0

    for sample_dir in sorted(DATA.glob("sample_*")):
        if copy_output_to_sample(sample_dir):
            count += 1

    print(f"Copied {count} LaMa outputs back into {DATA}")


if __name__ == "__main__":
    main()