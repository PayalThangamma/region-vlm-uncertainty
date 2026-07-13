import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

CONDITIONS = [
    "none",
    "all",
    "removed",
    "context",
    "background",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

    return rows


def main() -> None:
    reference_ids = None

    for condition in CONDITIONS:
        folder = OUTPUT_ROOT / f"eval_original_{condition}"
        captions_path = folder / "captions.jsonl"
        regions_path = folder / "region_uncertainty"

        if not captions_path.exists():
            raise FileNotFoundError(captions_path)

        if not regions_path.exists():
            raise FileNotFoundError(regions_path)

        rows = read_jsonl(captions_path)
        region_files = sorted(regions_path.glob("*.json"))

        if len(rows) != 522:
            raise ValueError(
                f"{condition}: expected 522 captions, found {len(rows)}"
            )

        if len(region_files) != 522:
            raise ValueError(
                f"{condition}: expected 522 region JSONs, "
                f"found {len(region_files)}"
            )

        sample_ids = [row["sample_id"] for row in rows]

        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(
                f"{condition}: duplicate sample IDs found"
            )

        if not all(
            str(row.get("label", "")).lower() == "yes"
            for row in rows
        ):
            raise ValueError(
                f"{condition}: not all labels are 'yes'"
            )

        current_ids = set(sample_ids)

        if reference_ids is None:
            reference_ids = current_ids
        elif current_ids != reference_ids:
            raise ValueError(
                f"{condition}: sample IDs differ from baseline"
            )

        print(
            f"{condition}: "
            f"{len(rows)} captions, "
            f"{len(region_files)} region JSONs, "
            "labels=yes"
        )

    print("\nOriginal-image sanity outputs verified successfully.")


if __name__ == "__main__":
    main()