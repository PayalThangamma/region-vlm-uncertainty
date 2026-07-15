import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAPTION_ROOT = (
    PROJECT_ROOT
    / "qwen_pipeline"
    / "outputs"
    / "eval_removed_full"
)

UNCERTAINTY_ROOT = (
    PROJECT_ROOT
    / "qwen_pipeline"
    / "outputs"
    / "uncertainty_removed_full"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "qwen25vl7b"
)

METRICS_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    "all",
    "removed",
    "context",
    "background",
]


def classify_answer(text: str) -> str:
    normalized = text.strip().lower()

    if normalized.startswith("yes"):
        return "yes"

    if normalized.startswith("no"):
        return "no"

    return "unknown"


def load_captions(condition: str) -> pd.DataFrame:
    path = CAPTION_ROOT / f"captions_{condition}.jsonl"

    if not path.exists():
        raise FileNotFoundError(path)

    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    frame = pd.DataFrame(records)

    if len(frame) != 522:
        raise ValueError(
            f"{condition}: expected 522 rows, found {len(frame)}"
        )

    if frame["sample_id"].duplicated().any():
        raise ValueError(
            f"{condition}: duplicate sample IDs found"
        )

    frame["answer_class"] = frame["text"].map(
        classify_answer
    )

    frame["hallucinated"] = (
        frame["answer_class"] == "yes"
    ).astype(int)

    return frame


def load_uncertainty() -> pd.DataFrame:
    records = []

    files = sorted(
        UNCERTAINTY_ROOT.glob("*.json")
    )

    if len(files) != 522:
        raise ValueError(
            f"Expected 522 uncertainty files, found {len(files)}"
        )

    for path in files:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        records.append(
            {
                "sample_id": data["sample_id"],
                "all_suppressed":
                    int(data["num_uncertain_tokens"]),
                "removed_suppressed":
                    int(
                        data["regions"]["removed"]["uncertain"]
                    ),
                "context_suppressed":
                    int(
                        data["regions"]["context"]["uncertain"]
                    ),
                "background_suppressed":
                    int(
                        data["regions"]["background"]["uncertain"]
                    ),
            }
        )

    frame = pd.DataFrame(records)

    if frame["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate uncertainty sample IDs found"
        )

    return frame


def main() -> None:
    baseline = load_captions("none")
    uncertainty = load_uncertainty()

    summary_rows = []
    sample_rows = []

    for condition in CONDITIONS:
        masked = load_captions(condition)

        merged = baseline.merge(
            masked,
            on="sample_id",
            how="inner",
            suffixes=("_none", f"_{condition}"),
            validate="one_to_one",
        )

        merged = merged.merge(
            uncertainty,
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )

        suppression_column = (
            f"{condition}_suppressed"
        )

        active = merged[
            merged[suppression_column] > 0
        ].copy()

        total_samples = len(merged)
        active_samples = len(active)

        none_hallucinated = int(
            active["hallucinated_none"].sum()
        )

        masked_hallucinated = int(
            active[
                f"hallucinated_{condition}"
            ].sum()
        )

        none_rate = (
            none_hallucinated / active_samples
            if active_samples
            else float("nan")
        )

        masked_rate = (
            masked_hallucinated / active_samples
            if active_samples
            else float("nan")
        )

        effect = none_rate - masked_rate

        yes_to_no = int(
            (
                (
                    active["answer_class_none"]
                    == "yes"
                )
                & (
                    active[
                        f"answer_class_{condition}"
                    ]
                    == "no"
                )
            ).sum()
        )

        no_to_yes = int(
            (
                (
                    active["answer_class_none"]
                    == "no"
                )
                & (
                    active[
                        f"answer_class_{condition}"
                    ]
                    == "yes"
                )
            ).sum()
        )

        summary_rows.append(
            {
                "model": "qwen25vl7b",
                "condition": condition,
                "total_samples": total_samples,
                "active_samples": active_samples,
                "active_fraction":
                    active_samples / total_samples,
                "none_hallucinated":
                    none_hallucinated,
                "masked_hallucinated":
                    masked_hallucinated,
                "none_rate": none_rate,
                "masked_rate": masked_rate,
                "effect": effect,
                "effect_pp": effect * 100,
                "yes_to_no": yes_to_no,
                "no_to_yes": no_to_yes,
                "net_hallucination_reduction":
                    yes_to_no - no_to_yes,
            }
        )

        for _, row in active.iterrows():
            sample_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "condition": condition,
                    "num_suppressed_tokens":
                        row[suppression_column],
                    "baseline_answer":
                        row["answer_class_none"],
                    "masked_answer":
                        row[
                            f"answer_class_{condition}"
                        ],
                    "baseline_hallucinated":
                        row["hallucinated_none"],
                    "masked_hallucinated":
                        row[
                            f"hallucinated_{condition}"
                        ],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    sample_df = pd.DataFrame(sample_rows)

    summary_path = (
        METRICS_DIR
        / "qwen_active_suppression_ablation.csv"
    )

    sample_path = (
        METRICS_DIR
        / "qwen_active_suppression_sample_level.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    sample_df.to_csv(sample_path, index=False)

    print("\nQwen active-suppression ablation")
    print(summary_df.to_string(index=False))

    print("\nSaved:")
    print(summary_path)
    print(sample_path)


if __name__ == "__main__":
    main()