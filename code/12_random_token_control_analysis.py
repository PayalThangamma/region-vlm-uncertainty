import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

METRICS_DIR = OUTPUT_ROOT / "metrics" / "llava7b"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "all": "random_all",
    "removed": "random_removed",
    "context": "random_context",
    "background": "random_background",
}

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


def classify_answer(text: str) -> str:
    """Classify a generated answer using the project yes/no rule."""
    normalized = text.strip().lower()

    if normalized.startswith("yes"):
        return "yes"

    if normalized.startswith("no"):
        return "no"

    return "unknown"


def load_captions(condition: str) -> pd.DataFrame:
    path = (
        OUTPUT_ROOT
        / f"eval_removed_{condition}"
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

    required_columns = {
        "sample_id",
        "question_id",
        "text",
        "label",
        "target_object",
        "prompt",
    }

    missing = required_columns.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    if frame["sample_id"].duplicated().any():
        duplicate_ids = frame.loc[
            frame["sample_id"].duplicated(),
            "sample_id",
        ].tolist()

        raise ValueError(
            f"Duplicate sample IDs in {path}: {duplicate_ids[:10]}"
        )

    frame["condition"] = condition
    frame["answer_class"] = frame["text"].map(classify_answer)
    frame["hallucinated"] = (
        frame["answer_class"] == "yes"
    ).astype(int)

    return frame


def paired_bootstrap(
    uncertainty_values: np.ndarray,
    random_values: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """
    Estimate uncertainty-minus-random improvement.

    Positive effect means uncertainty-guided masking has a lower
    hallucination rate than matched random masking.
    """
    if len(uncertainty_values) != len(random_values):
        raise ValueError("Paired arrays must have equal length.")

    n = len(uncertainty_values)

    if n == 0:
        raise ValueError("Cannot bootstrap an empty sample.")

    observed = float(
        random_values.mean() - uncertainty_values.mean()
    )

    paired_difference = (
        random_values - uncertainty_values
    )

    rng = np.random.default_rng(seed)

    bootstrap_effects = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for index in range(n_bootstrap):
        sampled_indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        bootstrap_effects[index] = paired_difference[
            sampled_indices
        ].mean()

    lower, upper = np.percentile(
        bootstrap_effects,
        [2.5, 97.5],
    )

    # Two-sided bootstrap sign test around zero.
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
        "uncertainty_advantage": observed,
        "uncertainty_advantage_pp": observed * 100,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_lower_pp": float(lower * 100),
        "ci_upper_pp": float(upper * 100),
        "p_value": float(p_value),
    }


def validate_pair(
    uncertainty: pd.DataFrame,
    random_control: pd.DataFrame,
    uncertainty_condition: str,
    random_condition: str,
) -> pd.DataFrame:
    uncertainty_ids = set(uncertainty["sample_id"])
    random_ids = set(random_control["sample_id"])

    if uncertainty_ids != random_ids:
        only_uncertainty = sorted(
            uncertainty_ids - random_ids
        )
        only_random = sorted(
            random_ids - uncertainty_ids
        )

        raise ValueError(
            f"Sample mismatch for {uncertainty_condition} vs "
            f"{random_condition}. "
            f"Only uncertainty: {only_uncertainty[:10]}; "
            f"only random: {only_random[:10]}"
        )

    merged = uncertainty.merge(
        random_control,
        on="sample_id",
        how="inner",
        suffixes=("_uncertainty", "_random"),
        validate="one_to_one",
    )

    if len(merged) != 522:
        raise ValueError(
            f"Expected 522 paired samples for "
            f"{uncertainty_condition} vs {random_condition}, "
            f"found {len(merged)}."
        )

    metadata_checks = [
        "question_id",
        "label",
        "target_object",
        "prompt",
    ]

    for column in metadata_checks:
        left = merged[f"{column}_uncertainty"]
        right = merged[f"{column}_random"]

        if not left.equals(right):
            raise ValueError(
                f"Metadata mismatch in column '{column}' for "
                f"{uncertainty_condition} vs {random_condition}."
            )

    return merged


def main() -> None:
    loaded = {}

    for uncertainty_condition, random_condition in PAIRS.items():
        for condition in [
            uncertainty_condition,
            random_condition,
        ]:
            if condition not in loaded:
                loaded[condition] = load_captions(condition)

    summary_rows = []
    comparison_rows = []
    flip_rows = []
    long_rows = []

    for uncertainty_condition, random_condition in PAIRS.items():
        uncertainty = loaded[uncertainty_condition]
        random_control = loaded[random_condition]

        merged = validate_pair(
            uncertainty,
            random_control,
            uncertainty_condition,
            random_condition,
        )

        uncertainty_values = merged[
            "hallucinated_uncertainty"
        ].to_numpy(dtype=np.int64)

        random_values = merged[
            "hallucinated_random"
        ].to_numpy(dtype=np.int64)

        uncertainty_yes = int(uncertainty_values.sum())
        random_yes = int(random_values.sum())

        uncertainty_no = int(
            len(uncertainty_values) - uncertainty_yes
        )
        random_no = int(
            len(random_values) - random_yes
        )

        uncertainty_unknown = int(
            (
                merged["answer_class_uncertainty"]
                == "unknown"
            ).sum()
        )

        random_unknown = int(
            (
                merged["answer_class_random"]
                == "unknown"
            ).sum()
        )

        uncertainty_rate = float(
            uncertainty_values.mean()
        )
        random_rate = float(random_values.mean())

        summary_rows.extend(
            [
                {
                    "condition_pair": uncertainty_condition,
                    "selection_strategy": "uncertainty",
                    "condition": uncertainty_condition,
                    "n": len(merged),
                    "hallucinated_yes": uncertainty_yes,
                    "correct_rejection_no": uncertainty_no,
                    "unknown": uncertainty_unknown,
                    "hallucination_rate": uncertainty_rate,
                    "hallucination_rate_percent":
                        uncertainty_rate * 100,
                },
                {
                    "condition_pair": uncertainty_condition,
                    "selection_strategy": "random_matched",
                    "condition": random_condition,
                    "n": len(merged),
                    "hallucinated_yes": random_yes,
                    "correct_rejection_no": random_no,
                    "unknown": random_unknown,
                    "hallucination_rate": random_rate,
                    "hallucination_rate_percent":
                        random_rate * 100,
                },
            ]
        )

        bootstrap = paired_bootstrap(
            uncertainty_values,
            random_values,
            seed=(
                BOOTSTRAP_SEED
                + list(PAIRS).index(
                    uncertainty_condition
                )
            ),
        )

        comparison_rows.append(
            {
                "uncertainty_condition":
                    uncertainty_condition,
                "random_condition": random_condition,
                "n": len(merged),
                "uncertainty_hallucination_rate":
                    uncertainty_rate,
                "random_hallucination_rate":
                    random_rate,
                "uncertainty_hallucination_rate_percent":
                    uncertainty_rate * 100,
                "random_hallucination_rate_percent":
                    random_rate * 100,
                **bootstrap,
            }
        )

        uncertainty_yes_random_no = int(
            (
                (
                    merged["answer_class_uncertainty"]
                    == "yes"
                )
                & (
                    merged["answer_class_random"]
                    == "no"
                )
            ).sum()
        )

        uncertainty_no_random_yes = int(
            (
                (
                    merged["answer_class_uncertainty"]
                    == "no"
                )
                & (
                    merged["answer_class_random"]
                    == "yes"
                )
            ).sum()
        )

        unchanged_yes = int(
            (
                (
                    merged["answer_class_uncertainty"]
                    == "yes"
                )
                & (
                    merged["answer_class_random"]
                    == "yes"
                )
            ).sum()
        )

        unchanged_no = int(
            (
                (
                    merged["answer_class_uncertainty"]
                    == "no"
                )
                & (
                    merged["answer_class_random"]
                    == "no"
                )
            ).sum()
        )

        unknown_changes = int(
            (
                (
                    merged["answer_class_uncertainty"]
                    == "unknown"
                )
                | (
                    merged["answer_class_random"]
                    == "unknown"
                )
            ).sum()
        )

        # Positive means uncertainty masking produces fewer
        # hallucinations than matched random masking.
        net_uncertainty_advantage = (
            uncertainty_no_random_yes
            - uncertainty_yes_random_no
        )

        flip_rows.append(
            {
                "uncertainty_condition":
                    uncertainty_condition,
                "random_condition": random_condition,
                "n": len(merged),
                "uncertainty_yes_random_no":
                    uncertainty_yes_random_no,
                "uncertainty_no_random_yes":
                    uncertainty_no_random_yes,
                "unchanged_yes": unchanged_yes,
                "unchanged_no": unchanged_no,
                "unknown_changes": unknown_changes,
                "net_uncertainty_advantage":
                    net_uncertainty_advantage,
            }
        )

        for _, row in merged.iterrows():
            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "target_object":
                        row["target_object_uncertainty"],
                    "prompt": row["prompt_uncertainty"],
                    "uncertainty_condition":
                        uncertainty_condition,
                    "random_condition": random_condition,
                    "uncertainty_text":
                        row["text_uncertainty"],
                    "random_text": row["text_random"],
                    "uncertainty_answer_class":
                        row["answer_class_uncertainty"],
                    "random_answer_class":
                        row["answer_class_random"],
                    "uncertainty_hallucinated":
                        row["hallucinated_uncertainty"],
                    "random_hallucinated":
                        row["hallucinated_random"],
                    "paired_advantage":
                        (
                            row["hallucinated_random"]
                            - row[
                                "hallucinated_uncertainty"
                            ]
                        ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    flips_df = pd.DataFrame(flip_rows)
    long_df = pd.DataFrame(long_rows)

    summary_path = (
        METRICS_DIR
        / "random_control_summary.csv"
    )

    comparison_path = (
        METRICS_DIR
        / "random_control_paired_comparison.csv"
    )

    flips_path = (
        METRICS_DIR
        / "random_control_answer_flips.csv"
    )

    long_path = (
        METRICS_DIR
        / "random_control_sample_level.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    flips_df.to_csv(flips_path, index=False)
    long_df.to_csv(long_path, index=False)

    print("\nMatched random-control summary")
    print(
        summary_df[
            [
                "condition_pair",
                "selection_strategy",
                "hallucinated_yes",
                "correct_rejection_no",
                "unknown",
                "hallucination_rate_percent",
            ]
        ].to_string(index=False)
    )

    print("\nPaired uncertainty-vs-random comparison")
    print(
        comparison_df[
            [
                "uncertainty_condition",
                "random_condition",
                "uncertainty_hallucination_rate_percent",
                "random_hallucination_rate_percent",
                "uncertainty_advantage_pp",
                "ci_lower_pp",
                "ci_upper_pp",
                "p_value",
            ]
        ].to_string(index=False)
    )

    print("\nAnswer differences")
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