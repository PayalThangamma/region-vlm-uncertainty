import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN_ROOT = PROJECT_ROOT / "qwen_pipeline" / "outputs" / "eval_removed_full"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics" / "qwen25vl7b"
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
    path = QWEN_ROOT / f"captions_{condition}.jsonl"

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
            f"{path} missing columns: {sorted(missing)}"
        )

    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")

    if len(frame) != 522:
        raise ValueError(
            f"{condition}: expected 522 rows, found {len(frame)}"
        )

    if not (frame["label"].str.lower() == "no").all():
        raise ValueError(
            f"{condition}: expected all labels to be 'no'"
        )

    frame["condition"] = condition
    frame["answer_class"] = frame["text"].map(classify_answer)
    frame["hallucinated"] = (
        frame["answer_class"] == "yes"
    ).astype(int)

    return frame


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
            f"{condition}: expected 522 paired rows, found {len(merged)}"
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
                f"{condition}: metadata mismatch in {column}"
            )

    return merged


def paired_bootstrap_effect(
    baseline_values: np.ndarray,
    masked_values: np.ndarray,
    seed: int,
) -> dict:
    # Positive effect means masking reduces hallucination.
    paired_effect = baseline_values - masked_values
    observed = float(paired_effect.mean())

    rng = np.random.default_rng(seed)
    n = len(paired_effect)

    bootstrap_effects = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=float,
    )

    for index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.integers(0, n, size=n)
        bootstrap_effects[index] = paired_effect[
            sampled
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
        "causal_effect": observed,
        "causal_effect_pp": observed * 100,
        "ci_lower_pp": float(lower * 100),
        "ci_upper_pp": float(upper * 100),
        "p_value": float(p_value),
    }


def main() -> None:
    frames = {
        condition: load_condition(condition)
        for condition in CONDITIONS
    }

    summary_rows = []
    bootstrap_rows = []
    flip_rows = []
    sample_rows = []

    for condition, frame in frames.items():
        hallucinated_yes = int(frame["hallucinated"].sum())
        correct_no = int(
            (frame["answer_class"] == "no").sum()
        )
        unknown = int(
            (frame["answer_class"] == "unknown").sum()
        )
        rate = float(frame["hallucinated"].mean())

        summary_rows.append(
            {
                "condition": condition,
                "n": len(frame),
                "hallucinated_yes": hallucinated_yes,
                "correct_rejection_no": correct_no,
                "unknown": unknown,
                "hallucination_rate": rate,
                "hallucination_rate_percent": rate * 100,
            }
        )

    baseline = frames["none"]

    for index, condition in enumerate(
        ["all", "removed", "context", "background"]
    ):
        merged = validate_pair(
            baseline,
            frames[condition],
            condition,
        )

        baseline_values = merged[
            "hallucinated_none"
        ].to_numpy(dtype=np.int64)

        masked_values = merged[
            f"hallucinated_{condition}"
        ].to_numpy(dtype=np.int64)

        bootstrap = paired_bootstrap_effect(
            baseline_values,
            masked_values,
            seed=BOOTSTRAP_SEED + index,
        )

        bootstrap_rows.append(
            {
                "condition": condition,
                "n": len(merged),
                "baseline_rate_percent":
                    baseline_values.mean() * 100,
                "masked_rate_percent":
                    masked_values.mean() * 100,
                **bootstrap,
            }
        )

        yes_to_no = int(
            (
                (merged["answer_class_none"] == "yes")
                & (
                    merged[f"answer_class_{condition}"]
                    == "no"
                )
            ).sum()
        )

        no_to_yes = int(
            (
                (merged["answer_class_none"] == "no")
                & (
                    merged[f"answer_class_{condition}"]
                    == "yes"
                )
            ).sum()
        )

        unchanged_yes = int(
            (
                (merged["answer_class_none"] == "yes")
                & (
                    merged[f"answer_class_{condition}"]
                    == "yes"
                )
            ).sum()
        )

        unchanged_no = int(
            (
                (merged["answer_class_none"] == "no")
                & (
                    merged[f"answer_class_{condition}"]
                    == "no"
                )
            ).sum()
        )

        unknown_changes = int(
            (
                (merged["answer_class_none"] == "unknown")
                | (
                    merged[f"answer_class_{condition}"]
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
                "net_hallucination_reduction":
                    yes_to_no - no_to_yes,
            }
        )

        for _, row in merged.iterrows():
            sample_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "target_object":
                        row["target_object_none"],
                    "condition": condition,
                    "baseline_text": row["text_none"],
                    "masked_text":
                        row[f"text_{condition}"],
                    "baseline_answer_class":
                        row["answer_class_none"],
                    "masked_answer_class":
                        row[f"answer_class_{condition}"],
                    "baseline_hallucinated":
                        row["hallucinated_none"],
                    "masked_hallucinated":
                        row[f"hallucinated_{condition}"],
                    "paired_effect":
                        row["hallucinated_none"]
                        - row[f"hallucinated_{condition}"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    flips_df = pd.DataFrame(flip_rows)
    sample_df = pd.DataFrame(sample_rows)

    summary_path = METRICS_DIR / "qwen_removed_summary.csv"
    bootstrap_path = (
        METRICS_DIR / "qwen_removed_bootstrap_effects.csv"
    )
    flips_path = METRICS_DIR / "qwen_removed_answer_flips.csv"
    sample_path = (
        METRICS_DIR / "qwen_removed_sample_level.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    flips_df.to_csv(flips_path, index=False)
    sample_df.to_csv(sample_path, index=False)

    print("\nQwen summary")
    print(summary_df.to_string(index=False))

    print("\nPaired bootstrap effects")
    print(bootstrap_df.to_string(index=False))

    print("\nAnswer flips")
    print(flips_df.to_string(index=False))

    print("\nSaved:")
    for path in [
        summary_path,
        bootstrap_path,
        flips_path,
        sample_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()