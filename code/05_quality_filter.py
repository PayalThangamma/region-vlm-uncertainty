from pathlib import Path
import json
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGION_MAP_DIR = PROJECT_ROOT / "outputs" / "region_maps_rohe"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "rohe_quality.csv"

MIN_BACKGROUND_TOKENS = 250
MIN_REMOVED_TOKENS = 20
MAX_REMOVED_TOKENS = 150
MIN_CONTEXT_TOKENS = 30
MAX_CONTEXT_TOKENS = 200


def is_good_sample(counts: dict) -> tuple[bool, list[str]]:
    reasons = []

    removed = counts["removed"]
    context = counts["context"]
    background = counts["background"]
    total = counts["total"]

    if total != 576:
        reasons.append("bad_total_token_count")

    if background < MIN_BACKGROUND_TOKENS:
        reasons.append("background_too_small")

    if removed < MIN_REMOVED_TOKENS:
        reasons.append("removed_too_small")

    if removed > MAX_REMOVED_TOKENS:
        reasons.append("removed_too_large")

    if context < MIN_CONTEXT_TOKENS:
        reasons.append("context_too_small")

    if context > MAX_CONTEXT_TOKENS:
        reasons.append("context_too_large")

    return len(reasons) == 0, reasons


def main():
    if not REGION_MAP_DIR.exists():
        raise FileNotFoundError(f"Region map directory not found: {REGION_MAP_DIR}")

    sample_dirs = sorted(
        [p for p in REGION_MAP_DIR.iterdir() if p.is_dir() and p.name.startswith("sample_")]
    )

    rows = []
    good_count = 0

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        counts_path = sample_dir / "region_counts.json"

        if not counts_path.exists():
            rows.append({
                "sample_id": sample_id,
                "removed": "",
                "context": "",
                "background": "",
                "total": "",
                "good_sample": False,
                "reasons": "missing_region_counts",
            })
            continue

        with open(counts_path, "r", encoding="utf-8") as f:
            counts = json.load(f)

        good, reasons = is_good_sample(counts)

        if good:
            good_count += 1

        rows.append({
            "sample_id": sample_id,
            "removed": counts["removed"],
            "context": counts["context"],
            "background": counts["background"],
            "total": counts["total"],
            "good_sample": good,
            "reasons": ";".join(reasons),
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "removed",
                "context",
                "background",
                "total",
                "good_sample",
                "reasons",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Samples checked: {len(rows)}")
    print(f"Good samples: {good_count} / {len(rows)}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()