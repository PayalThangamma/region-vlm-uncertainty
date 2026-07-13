import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_ROOT / "metrics" / "llava7b"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    "none",
    "all",
    "removed",
    "context",
    "background",
]

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


def classify_answer(text: str) -> str:
    normalized = text.strip().lower()

    if normalized.startswith("yes"):
        return "yes"
    if normalized.startswith("no"):
        return "no"
    return "unknown"


def load_condition(condition: str) -> pd.DataFrame:
    path = (
        OUTPUT_ROOT
        / f"eval_original_{condition}"
        / "captions.jsonl"
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing captions file: {path}")

    records: List[Dict] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

            records.append(record)

    frame = pd.DataFrame(records)

    required = {
        "sample_id",
        "question_id",
        "text",
        "label",
        "target_object",
        "prompt",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")

    if not (frame["label"].str.lower() == "yes").all():
        raise ValueError(
            f"Expected all original-image labels to be 'yes' in {path}"
        )

    frame["condition"] = condition
    frame["answer_class"] = frame["text"].map(classify_answer)

    frame["correct"] = (
        frame["answer_class"] == "yes"
    ).astype(int)

    frame["false_negative"] = (
        frame["answer_class"] == "no"
    ).astype(int)

    return frame


def paired_bootstrap_accuracy_drop(
    baseline_correct: np.ndarray,
    masked_correct: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Positive drop means masking reduces original-image accuracy.
    """
    if len(baseline_correct) != len(masked_correct):
        raise ValueError("Paired arrays must have equal length.")

    paired_drop = baseline_correct - masked_correct
    observed = float(paired_drop.mean())

    rng = np.random.default_rng(seed)
    n = len(paired_drop)

    bootstrap_effects = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        sampled_indices = rng.integers(0, n, size=n)
        bootstrap_effects[index] = paired_drop[
            sampled_indices
        ].mean()

    lower, upper = np.percentile(
        bootstrap_effects,
        [2.5, 97.5],
    )

    probability_non_positive = np.mean(
        bootstrap_effects <= 0
    )
    probability_non_negative = np.mean(
        bootstrap_effects >= 0
    )

    p_value = min(
        1.0,
        2.0
        * min(
            probability_non_positive,
            probability_non_negative,
        ),
    )

    return {
        "accuracy_drop": observed,
        "accuracy_drop_pp": observed * 100,
        "ci_lower_pp": float(lower * 100),
        "ci_upper_pp": float(upper * 100),
        "p_value": float(p_value),
    }


def validate_pair(
    baseline: pd.DataFrame,
    masked: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    merged = baseline.merge(
        masked,
        on="sample_id",
        how="inner",
        suffixes=("_none", f"_{condition}"),
        validate="one_to_one",
    )

    if len(merged) != 522:
        raise ValueError(
            f"Expected 522 paired samples for {condition}, "
            f"found {len(merged)}"
        )

    for column in [
        "question_id",
        "label",
        "target_object",
        "prompt",
    ]:
        left = merged[f"{column}_none"]
        right = merged[f"{column}_{condition}"]

        if not left.equals(right):
            raise ValueError(
                f"Metadata mismatch in {column} for {condition}"
            )

    return merged


def main() -> None:
    frames = {
        condition: load_condition(condition)
        for condition in CONDITIONS
    }

    summary_rows = []
    comparison_rows = []
    flip_rows = []
    long_rows = []

    for condition, frame in frames.items():
        total = len(frame)
        correct_yes = int(frame["correct"].sum())
        false_negative_no = int(
            frame["false_negative"].sum()
        )
        unknown = int(
            (frame["answer_class"] == "unknown").sum()
        )

        accuracy = float(frame["correct"].mean())
        false_negative_rate = float(
            frame["false_negative"].mean()
        )

        summary_rows.append(
            {
                "condition": condition,
                "n": total,
                "correct_yes": correct_yes,
                "false_negative_no": false_negative_no,
                "unknown": unknown,
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100,
                "false_negative_rate":
                    false_negative_rate,
                "false_negative_rate_percent":
                    false_negative_rate * 100,
            }
        )

    baseline = frames["none"]

    for condition in [
        "all",
        "removed",
        "context",
        "background",
    ]:
        masked = frames[condition]

        merged = validate_pair(
            baseline,
            masked,
            condition,
        )

        baseline_correct = merged[
            "correct_none"
        ].to_numpy(dtype=np.int64)

        masked_correct = merged[
            f"correct_{condition}"
        ].to_numpy(dtype=np.int64)

        bootstrap = paired_bootstrap_accuracy_drop(
            baseline_correct,
            masked_correct,
            seed=(
                BOOTSTRAP_SEED
                + [
                    "all",
                    "removed",
                    "context",
                    "background",
                ].index(condition)
            ),
        )

        comparison_rows.append(
            {
                "condition": condition,
                "n": len(merged),
                "baseline_accuracy_percent":
                    baseline_correct.mean() * 100,
                "masked_accuracy_percent":
                    masked_correct.mean() * 100,
                **bootstrap,
            }
        )

        yes_to_no = int(
            (
                (
                    merged["answer_class_none"]
                    == "yes"
                )
                & (
                    merged[
                        f"answer_class_{condition}"
                    ]
                    == "no"
                )
            ).sum()
        )

        no_to_yes = int(
            (
                (
                    merged["answer_class_none"]
                    == "no"
                )
                & (
                    merged[
                        f"answer_class_{condition}"
                    ]
                    == "yes"
                )
            ).sum()
        )

        unchanged_yes = int(
            (
                (
                    merged["answer_class_none"]
                    == "yes"
                )
                & (
                    merged[
                        f"answer_class_{condition}"
                    ]
                    == "yes"
                )
            ).sum()
        )

        unchanged_no = int(
            (
                (
                    merged["answer_class_none"]
                    == "no"
                )
                & (
                    merged[
                        f"answer_class_{condition}"
                    ]
                    == "no"
                )
            ).sum()
        )

        unknown_changes = int(
            (
                (
                    merged["answer_class_none"]
                    == "unknown"
                )
                | (
                    merged[
                        f"answer_class_{condition}"
                    ]
                    == "unknown"
                )
            ).sum()
        )

        flip_rows.append(
            {
                "condition": condition,
                "n": len(merged),
                "yes_to_no": yes_to_no,
                "no_to_yes": no_to_yes,
                "unchanged_yes": unchanged_yes,
                "unchanged_no": unchanged_no,
                "unknown_changes": unknown_changes,
                "net_accuracy_change":
                    no_to_yes - yes_to_no,
            }
        )

        for _, row in merged.iterrows():
            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "target_object":
                        row["target_object_none"],
                    "prompt": row["prompt_none"],
                    "condition": condition,
                    "baseline_text": row["text_none"],
                    "masked_text":
                        row[f"text_{condition}"],
                    "baseline_answer_class":
                        row["answer_class_none"],
                    "masked_answer_class":
                        row[
                            f"answer_class_{condition}"
                        ],
                    "baseline_correct":
                        row["correct_none"],
                    "masked_correct":
                        row[f"correct_{condition}"],
                    "accuracy_change":
                        (
                            row[f"correct_{condition}"]
                            - row["correct_none"]
                        ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    flips_df = pd.DataFrame(flip_rows)
    long_df = pd.DataFrame(long_rows)

    summary_path = (
        METRICS_DIR
        / "original_sanity_summary.csv"
    )

    comparison_path = (
        METRICS_DIR
        / "original_sanity_paired_bootstrap.csv"
    )

    flips_path = (
        METRICS_DIR
        / "original_sanity_answer_flips.csv"
    )

    long_path = (
        METRICS_DIR
        / "original_sanity_sample_level.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    flips_df.to_csv(flips_path, index=False)
    long_df.to_csv(long_path, index=False)

    print("\nOriginal-image sanity summary")
    print(summary_df.to_string(index=False))

    print("\nPaired accuracy-drop analysis")
    print(comparison_df.to_string(index=False))

    print("\nAnswer flips")
    print(flips_df.to_string(index=False))

    print("\nSaved:")
    for path in [
        summary_path,
        comparison_path,
        flips_path,
        long_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()