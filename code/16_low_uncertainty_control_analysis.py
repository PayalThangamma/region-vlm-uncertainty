import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_ROOT / "metrics" / "llava7b"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = [
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


def load_condition(folder_name: str) -> pd.DataFrame:
    path = OUTPUT_ROOT / folder_name / "captions.jsonl"

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

    frame["answer_class"] = frame["text"].map(classify_answer)
    frame["hallucinated"] = (
        frame["answer_class"] == "yes"
    ).astype(int)

    return frame


def validate_pair(
    high: pd.DataFrame,
    comparison: pd.DataFrame,
    region: str,
    comparison_name: str,
) -> pd.DataFrame:
    merged = high.merge(
        comparison,
        on="sample_id",
        how="inner",
        suffixes=("_high", f"_{comparison_name}"),
        validate="one_to_one",
    )

    if len(merged) != 522:
        raise ValueError(
            f"{region}, {comparison_name}: expected 522 pairs, "
            f"found {len(merged)}"
        )

    for column in [
        "question_id",
        "label",
        "target_object",
        "prompt",
    ]:
        left = merged[f"{column}_high"]
        right = merged[
            f"{column}_{comparison_name}"
        ]

        if not left.equals(right):
            raise ValueError(
                f"{region}, {comparison_name}: "
                f"metadata mismatch in {column}"
            )

    return merged


def paired_bootstrap_advantage(
    high_values: np.ndarray,
    comparison_values: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Positive advantage means high-uncertainty masking
    has a lower hallucination rate than comparison masking.
    """
    paired_difference = comparison_values - high_values
    observed = float(paired_difference.mean())

    rng = np.random.default_rng(seed)
    n = len(paired_difference)

    bootstrap_effects = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        sampled_indices = rng.integers(0, n, size=n)
        bootstrap_effects[index] = paired_difference[
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
        "high_uncertainty_advantage":
            observed,
        "high_uncertainty_advantage_pp":
            observed * 100,
        "ci_lower_pp": float(lower * 100),
        "ci_upper_pp": float(upper * 100),
        "p_value": float(p_value),
    }


def main() -> None:
    summary_rows = []
    comparison_rows = []
    flip_rows = []
    long_rows = []

    for region in REGIONS:
        high_folder = f"eval_removed_{region}"
        low_folder = f"eval_removed_low_{region}"
        random_folder = f"eval_removed_random_{region}"

        high = load_condition(high_folder)
        low = load_condition(low_folder)
        random_control = load_condition(random_folder)

        strategies = [
            ("high_uncertainty", high),
            ("matched_random_seed42", random_control),
            ("matched_low_uncertainty", low),
        ]

        for strategy, frame in strategies:
            hallucinated_yes = int(
                frame["hallucinated"].sum()
            )
            correct_no = int(
                len(frame) - hallucinated_yes
            )
            unknown = int(
                (
                    frame["answer_class"] == "unknown"
                ).sum()
            )
            rate = float(
                frame["hallucinated"].mean()
            )

            summary_rows.append(
                {
                    "region": region,
                    "strategy": strategy,
                    "n": len(frame),
                    "hallucinated_yes":
                        hallucinated_yes,
                    "correct_rejection_no":
                        correct_no,
                    "unknown": unknown,
                    "hallucination_rate": rate,
                    "hallucination_rate_percent":
                        rate * 100,
                }
            )

        for comparison_name, comparison in [
            ("low", low),
            ("random", random_control),
        ]:
            merged = validate_pair(
                high,
                comparison,
                region,
                comparison_name,
            )

            high_values = merged[
                "hallucinated_high"
            ].to_numpy(dtype=np.int64)

            comparison_values = merged[
                f"hallucinated_{comparison_name}"
            ].to_numpy(dtype=np.int64)

            bootstrap = paired_bootstrap_advantage(
                high_values,
                comparison_values,
                seed=(
                    BOOTSTRAP_SEED
                    + REGIONS.index(region) * 10
                    + (
                        0
                        if comparison_name == "low"
                        else 1
                    )
                ),
            )

            comparison_rows.append(
                {
                    "region": region,
                    "comparison": comparison_name,
                    "n": len(merged),
                    "high_rate_percent":
                        high_values.mean() * 100,
                    "comparison_rate_percent":
                        comparison_values.mean() * 100,
                    **bootstrap,
                }
            )

            high_yes_comparison_no = int(
                (
                    (
                        merged[
                            "answer_class_high"
                        ]
                        == "yes"
                    )
                    & (
                        merged[
                            f"answer_class_{comparison_name}"
                        ]
                        == "no"
                    )
                ).sum()
            )

            high_no_comparison_yes = int(
                (
                    (
                        merged[
                            "answer_class_high"
                        ]
                        == "no"
                    )
                    & (
                        merged[
                            f"answer_class_{comparison_name}"
                        ]
                        == "yes"
                    )
                ).sum()
            )

            unchanged_yes = int(
                (
                    (
                        merged[
                            "answer_class_high"
                        ]
                        == "yes"
                    )
                    & (
                        merged[
                            f"answer_class_{comparison_name}"
                        ]
                        == "yes"
                    )
                ).sum()
            )

            unchanged_no = int(
                (
                    (
                        merged[
                            "answer_class_high"
                        ]
                        == "no"
                    )
                    & (
                        merged[
                            f"answer_class_{comparison_name}"
                        ]
                        == "no"
                    )
                ).sum()
            )

            unknown_changes = int(
                (
                    (
                        merged[
                            "answer_class_high"
                        ]
                        == "unknown"
                    )
                    | (
                        merged[
                            f"answer_class_{comparison_name}"
                        ]
                        == "unknown"
                    )
                ).sum()
            )

            flip_rows.append(
                {
                    "region": region,
                    "comparison": comparison_name,
                    "n": len(merged),
                    "high_yes_comparison_no":
                        high_yes_comparison_no,
                    "high_no_comparison_yes":
                        high_no_comparison_yes,
                    "unchanged_yes":
                        unchanged_yes,
                    "unchanged_no":
                        unchanged_no,
                    "unknown_changes":
                        unknown_changes,
                    "net_high_uncertainty_advantage":
                        (
                            high_no_comparison_yes
                            - high_yes_comparison_no
                        ),
                }
            )

            for _, row in merged.iterrows():
                long_rows.append(
                    {
                        "sample_id":
                            row["sample_id"],
                        "target_object":
                            row[
                                "target_object_high"
                            ],
                        "region": region,
                        "comparison":
                            comparison_name,
                        "high_text":
                            row["text_high"],
                        "comparison_text":
                            row[
                                f"text_{comparison_name}"
                            ],
                        "high_answer_class":
                            row[
                                "answer_class_high"
                            ],
                        "comparison_answer_class":
                            row[
                                f"answer_class_{comparison_name}"
                            ],
                        "high_hallucinated":
                            row[
                                "hallucinated_high"
                            ],
                        "comparison_hallucinated":
                            row[
                                f"hallucinated_{comparison_name}"
                            ],
                        "paired_advantage":
                            (
                                row[
                                    f"hallucinated_{comparison_name}"
                                ]
                                - row[
                                    "hallucinated_high"
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
        / "low_uncertainty_control_summary.csv"
    )

    comparison_path = (
        METRICS_DIR
        / "low_uncertainty_control_paired_bootstrap.csv"
    )

    flips_path = (
        METRICS_DIR
        / "low_uncertainty_control_answer_flips.csv"
    )

    long_path = (
        METRICS_DIR
        / "low_uncertainty_control_sample_level.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    flips_df.to_csv(flips_path, index=False)
    long_df.to_csv(long_path, index=False)

    print("\nHigh vs random vs low summary")
    print(summary_df.to_string(index=False))

    print("\nPaired comparisons")
    print(comparison_df.to_string(index=False))

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