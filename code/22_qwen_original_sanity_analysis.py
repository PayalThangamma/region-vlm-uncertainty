import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_ROOT = (
    PROJECT_ROOT
    / "qwen_pipeline"
    / "outputs"
    / "eval_original_full"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "qwen25vl7b"
)

METRICS_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["none", "all", "removed", "context", "background"]
N_BOOTSTRAP = 10_000
BASE_SEED = 42
EXPECTED_ROWS = 522


def classify_answer(text: str) -> str:
    normalized = str(text).strip().lower()
    if normalized.startswith("yes"):
        return "yes"
    if normalized.startswith("no"):
        return "no"
    return "unknown"


def load_condition(condition: str) -> pd.DataFrame:
    path = EVAL_ROOT / f"captions_{condition}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

    frame = pd.DataFrame(records)
    required_columns = {"sample_id", "text"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"{condition}: expected {EXPECTED_ROWS} rows, found {len(frame)}"
        )

    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[
            frame["sample_id"].duplicated(), "sample_id"
        ].tolist()
        raise ValueError(
            f"{condition}: duplicate sample IDs found: {duplicates[:10]}"
        )

    if "label" in frame.columns:
        labels = frame["label"].astype(str).str.lower()
        if not (labels == "yes").all():
            raise ValueError(
                f"{condition}: expected all original-image labels to be 'yes', "
                f"found {sorted(labels.unique().tolist())}"
            )

    frame["answer_class"] = frame["text"].map(classify_answer)
    frame["correct"] = (frame["answer_class"] == "yes").astype(int)
    frame["false_negative"] = (frame["answer_class"] == "no").astype(int)
    return frame


def validate_pair(
    baseline: pd.DataFrame,
    masked: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    baseline_ids = set(baseline["sample_id"])
    masked_ids = set(masked["sample_id"])
    if baseline_ids != masked_ids:
        missing = sorted(baseline_ids - masked_ids)
        extra = sorted(masked_ids - baseline_ids)
        raise ValueError(
            f"{condition}: sample IDs do not match. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )

    columns = [
        "sample_id",
        "text",
        "answer_class",
        "correct",
        "false_negative",
    ]

    merged = baseline[columns].merge(
        masked[columns],
        on="sample_id",
        how="inner",
        suffixes=("_none", f"_{condition}"),
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_ROWS:
        raise ValueError(
            f"{condition}: expected {EXPECTED_ROWS} paired rows, "
            f"found {len(merged)}"
        )
    return merged


def paired_bootstrap_accuracy_drop(
    baseline_correct: np.ndarray,
    masked_correct: np.ndarray,
    seed: int,
) -> dict:
    """Positive accuracy_drop means masking reduced original-image accuracy."""
    if len(baseline_correct) != len(masked_correct):
        raise ValueError("Paired arrays have different lengths.")

    paired_drop = baseline_correct - masked_correct
    observed_drop = float(paired_drop.mean())
    rng = np.random.default_rng(seed)
    n = len(paired_drop)
    bootstrap_drops = np.empty(N_BOOTSTRAP, dtype=float)

    for index in range(N_BOOTSTRAP):
        sampled_indices = rng.integers(0, n, size=n)
        bootstrap_drops[index] = paired_drop[sampled_indices].mean()

    lower, upper = np.percentile(bootstrap_drops, [2.5, 97.5])
    p_non_positive = float(np.mean(bootstrap_drops <= 0))
    p_non_negative = float(np.mean(bootstrap_drops >= 0))
    p_value = min(1.0, 2.0 * min(p_non_positive, p_non_negative))

    return {
        "accuracy_drop": observed_drop,
        "accuracy_drop_pp": observed_drop * 100,
        "ci_lower_pp": float(lower * 100),
        "ci_upper_pp": float(upper * 100),
        "p_value": float(p_value),
    }


def main() -> None:
    frames = {condition: load_condition(condition) for condition in CONDITIONS}
    baseline = frames["none"]

    summary_rows = []
    paired_rows = []
    flip_rows = []
    sample_rows = []

    for condition in CONDITIONS:
        frame = frames[condition]
        correct_yes = int(frame["correct"].sum())
        false_negative_no = int(frame["false_negative"].sum())
        unknown = int((frame["answer_class"] == "unknown").sum())
        accuracy = float(frame["correct"].mean())
        false_negative_rate = float(frame["false_negative"].mean())

        summary_rows.append(
            {
                "condition": condition,
                "n": len(frame),
                "correct_yes": correct_yes,
                "false_negative_no": false_negative_no,
                "unknown": unknown,
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100,
                "false_negative_rate": false_negative_rate,
                "false_negative_rate_percent": false_negative_rate * 100,
            }
        )

    for index, condition in enumerate(["all", "removed", "context", "background"]):
        merged = validate_pair(baseline, frames[condition], condition)
        baseline_correct = merged["correct_none"].to_numpy(dtype=np.int64)
        masked_correct = merged[f"correct_{condition}"].to_numpy(dtype=np.int64)

        bootstrap = paired_bootstrap_accuracy_drop(
            baseline_correct,
            masked_correct,
            seed=BASE_SEED + index,
        )

        paired_rows.append(
            {
                "condition": condition,
                "n": len(merged),
                "baseline_accuracy_percent": float(
                    baseline_correct.mean() * 100
                ),
                "masked_accuracy_percent": float(masked_correct.mean() * 100),
                **bootstrap,
            }
        )

        yes_to_no = int(
            (
                (merged["answer_class_none"] == "yes")
                & (merged[f"answer_class_{condition}"] == "no")
            ).sum()
        )
        no_to_yes = int(
            (
                (merged["answer_class_none"] == "no")
                & (merged[f"answer_class_{condition}"] == "yes")
            ).sum()
        )
        unchanged_yes = int(
            (
                (merged["answer_class_none"] == "yes")
                & (merged[f"answer_class_{condition}"] == "yes")
            ).sum()
        )
        unchanged_no = int(
            (
                (merged["answer_class_none"] == "no")
                & (merged[f"answer_class_{condition}"] == "no")
            ).sum()
        )
        unknown_changes = int(
            (
                (merged["answer_class_none"] == "unknown")
                | (merged[f"answer_class_{condition}"] == "unknown")
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
                "net_accuracy_change": no_to_yes - yes_to_no,
            }
        )

        for _, row in merged.iterrows():
            sample_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "condition": condition,
                    "baseline_text": row["text_none"],
                    "masked_text": row[f"text_{condition}"],
                    "baseline_answer_class": row["answer_class_none"],
                    "masked_answer_class": row[f"answer_class_{condition}"],
                    "baseline_correct": row["correct_none"],
                    "masked_correct": row[f"correct_{condition}"],
                    "accuracy_change": (
                        row[f"correct_{condition}"] - row["correct_none"]
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    paired_df = pd.DataFrame(paired_rows)
    flips_df = pd.DataFrame(flip_rows)
    sample_df = pd.DataFrame(sample_rows)

    summary_path = METRICS_DIR / "qwen_original_sanity_summary.csv"
    paired_path = METRICS_DIR / "qwen_original_sanity_paired_bootstrap.csv"
    flips_path = METRICS_DIR / "qwen_original_sanity_answer_flips.csv"
    sample_path = METRICS_DIR / "qwen_original_sanity_sample_level.csv"

    summary_df.to_csv(summary_path, index=False)
    paired_df.to_csv(paired_path, index=False)
    flips_df.to_csv(flips_path, index=False)
    sample_df.to_csv(sample_path, index=False)

    print("\nQwen original-image sanity summary")
    print(summary_df.to_string(index=False))
    print("\nPaired accuracy-drop analysis")
    print(paired_df.to_string(index=False))
    print("\nAnswer flips")
    print(flips_df.to_string(index=False))
    print("\nSaved:")
    for path in [summary_path, paired_path, flips_path, sample_path]:
        print(path)


if __name__ == "__main__":
    main()
