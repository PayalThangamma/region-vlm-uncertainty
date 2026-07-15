import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HIGH_ROOT = (
    PROJECT_ROOT
    / "qwen_pipeline"
    / "outputs"
    / "eval_removed_full"
)

RANDOM_ROOT = (
    PROJECT_ROOT
    / "qwen_pipeline"
    / "outputs"
    / "eval_removed_random_seed42"
)

LOW_ROOT = (
    PROJECT_ROOT
    / "qwen_pipeline"
    / "outputs"
    / "eval_removed_low_full"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "qwen25vl7b"
)

METRICS_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = [
    "all",
    "removed",
    "context",
    "background",
]

N_BOOTSTRAP = 10_000
BASE_SEED = 42


def classify_answer(text: str) -> str:
    normalized = str(text).strip().lower()

    if normalized.startswith("yes"):
        return "yes"

    if normalized.startswith("no"):
        return "no"

    return "unknown"


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}"
                ) from exc

    frame = pd.DataFrame(records)

    required_columns = {
        "sample_id",
        "text",
    }

    missing = required_columns.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    if len(frame) != 522:
        raise ValueError(
            f"{path}: expected 522 rows, "
            f"found {len(frame)}"
        )

    if frame["sample_id"].duplicated().any():
        duplicates = (
            frame.loc[
                frame["sample_id"].duplicated(),
                "sample_id",
            ]
            .tolist()
        )

        raise ValueError(
            f"{path}: duplicate sample IDs found: "
            f"{duplicates[:10]}"
        )

    frame["answer_class"] = frame["text"].map(
        classify_answer
    )

    frame["hallucinated"] = (
        frame["answer_class"] == "yes"
    ).astype(int)

    return frame


def validate_sample_ids(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    name: str,
) -> None:
    reference_ids = set(reference["sample_id"])
    comparison_ids = set(comparison["sample_id"])

    if reference_ids != comparison_ids:
        missing = sorted(
            reference_ids - comparison_ids
        )

        extra = sorted(
            comparison_ids - reference_ids
        )

        raise ValueError(
            f"{name}: sample IDs do not match. "
            f"Missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )


def paired_bootstrap(
    high_values: np.ndarray,
    comparison_values: np.ndarray,
    seed: int,
) -> dict:
    """
    Positive advantage means high-uncertainty masking
    has a lower hallucination rate than the comparison.
    """

    if len(high_values) != len(comparison_values):
        raise ValueError(
            "Paired arrays have different lengths."
        )

    paired_advantage = (
        comparison_values - high_values
    )

    observed = float(
        paired_advantage.mean()
    )

    rng = np.random.default_rng(seed)

    n = len(paired_advantage)

    bootstrap_effects = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    for index in range(N_BOOTSTRAP):
        sampled_indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        bootstrap_effects[index] = (
            paired_advantage[
                sampled_indices
            ].mean()
        )

    lower, upper = np.percentile(
        bootstrap_effects,
        [2.5, 97.5],
    )

    probability_non_positive = float(
        np.mean(
            bootstrap_effects <= 0
        )
    )

    probability_non_negative = float(
        np.mean(
            bootstrap_effects >= 0
        )
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
        "ci_lower_pp":
            float(lower * 100),
        "ci_upper_pp":
            float(upper * 100),
        "p_value":
            float(p_value),
    }


def count_answer_differences(
    merged: pd.DataFrame,
    comparison_name: str,
) -> dict:
    high_class_column = (
        "answer_class_high"
    )

    comparison_class_column = (
        f"answer_class_{comparison_name}"
    )

    high_yes_comparison_no = int(
        (
            (
                merged[high_class_column]
                == "yes"
            )
            & (
                merged[
                    comparison_class_column
                ]
                == "no"
            )
        ).sum()
    )

    high_no_comparison_yes = int(
        (
            (
                merged[high_class_column]
                == "no"
            )
            & (
                merged[
                    comparison_class_column
                ]
                == "yes"
            )
        ).sum()
    )

    unchanged_yes = int(
        (
            (
                merged[high_class_column]
                == "yes"
            )
            & (
                merged[
                    comparison_class_column
                ]
                == "yes"
            )
        ).sum()
    )

    unchanged_no = int(
        (
            (
                merged[high_class_column]
                == "no"
            )
            & (
                merged[
                    comparison_class_column
                ]
                == "no"
            )
        ).sum()
    )

    unknown_changes = int(
        (
            (
                merged[high_class_column]
                == "unknown"
            )
            | (
                merged[
                    comparison_class_column
                ]
                == "unknown"
            )
        ).sum()
    )

    net_high_uncertainty_advantage = (
        high_no_comparison_yes
        - high_yes_comparison_no
    )

    return {
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
            net_high_uncertainty_advantage,
    }


def main() -> None:
    summary_rows = []
    paired_rows = []
    answer_difference_rows = []
    sample_rows = []

    for region_index, region in enumerate(
        REGIONS
    ):
        high_path = (
            HIGH_ROOT
            / f"captions_{region}.jsonl"
        )

        random_path = (
            RANDOM_ROOT
            / f"captions_random_{region}.jsonl"
        )

        low_path = (
            LOW_ROOT
            / f"captions_low_{region}.jsonl"
        )

        high = load_jsonl(high_path)
        random = load_jsonl(random_path)
        low = load_jsonl(low_path)

        validate_sample_ids(
            high,
            random,
            f"{region} random",
        )

        validate_sample_ids(
            high,
            low,
            f"{region} low",
        )

        high_rate = float(
            high["hallucinated"].mean()
        )

        random_rate = float(
            random["hallucinated"].mean()
        )

        low_rate = float(
            low["hallucinated"].mean()
        )

        for strategy, frame, rate in [
            (
                "high_uncertainty",
                high,
                high_rate,
            ),
            (
                "matched_random_seed42",
                random,
                random_rate,
            ),
            (
                "matched_low_uncertainty",
                low,
                low_rate,
            ),
        ]:
            hallucinated_yes = int(
                frame["hallucinated"].sum()
            )

            correct_rejection_no = int(
                (
                    frame["answer_class"]
                    == "no"
                ).sum()
            )

            unknown = int(
                (
                    frame["answer_class"]
                    == "unknown"
                ).sum()
            )

            summary_rows.append(
                {
                    "region": region,
                    "strategy": strategy,
                    "n": len(frame),
                    "hallucinated_yes":
                        hallucinated_yes,
                    "correct_rejection_no":
                        correct_rejection_no,
                    "unknown":
                        unknown,
                    "hallucination_rate":
                        rate,
                    "hallucination_rate_percent":
                        rate * 100,
                }
            )

        high_for_merge = high[
            [
                "sample_id",
                "text",
                "answer_class",
                "hallucinated",
            ]
        ].copy()

        random_for_merge = random[
            [
                "sample_id",
                "text",
                "answer_class",
                "hallucinated",
            ]
        ].copy()

        low_for_merge = low[
            [
                "sample_id",
                "text",
                "answer_class",
                "hallucinated",
            ]
        ].copy()

        high_random = high_for_merge.merge(
            random_for_merge,
            on="sample_id",
            how="inner",
            suffixes=(
                "_high",
                "_random",
            ),
            validate="one_to_one",
        )

        high_low = high_for_merge.merge(
            low_for_merge,
            on="sample_id",
            how="inner",
            suffixes=(
                "_high",
                "_low",
            ),
            validate="one_to_one",
        )

        if len(high_random) != 522:
            raise ValueError(
                f"{region}: high-random merge "
                f"returned {len(high_random)} rows"
            )

        if len(high_low) != 522:
            raise ValueError(
                f"{region}: high-low merge "
                f"returned {len(high_low)} rows"
            )

        high_values_random = (
            high_random[
                "hallucinated_high"
            ]
            .to_numpy(dtype=np.int64)
        )

        random_values = (
            high_random[
                "hallucinated_random"
            ]
            .to_numpy(dtype=np.int64)
        )

        random_bootstrap = paired_bootstrap(
            high_values=high_values_random,
            comparison_values=random_values,
            seed=(
                BASE_SEED
                + region_index
            ),
        )

        paired_rows.append(
            {
                "region": region,
                "comparison": "random",
                "n": len(high_random),
                "high_rate_percent":
                    float(
                        high_values_random.mean()
                        * 100
                    ),
                "comparison_rate_percent":
                    float(
                        random_values.mean()
                        * 100
                    ),
                **random_bootstrap,
            }
        )

        random_differences = (
            count_answer_differences(
                merged=high_random,
                comparison_name="random",
            )
        )

        answer_difference_rows.append(
            {
                "region": region,
                "comparison": "random",
                "n": len(high_random),
                **random_differences,
            }
        )

        high_values_low = (
            high_low[
                "hallucinated_high"
            ]
            .to_numpy(dtype=np.int64)
        )

        low_values = (
            high_low[
                "hallucinated_low"
            ]
            .to_numpy(dtype=np.int64)
        )

        low_bootstrap = paired_bootstrap(
            high_values=high_values_low,
            comparison_values=low_values,
            seed=(
                BASE_SEED
                + 100
                + region_index
            ),
        )

        paired_rows.append(
            {
                "region": region,
                "comparison": "low",
                "n": len(high_low),
                "high_rate_percent":
                    float(
                        high_values_low.mean()
                        * 100
                    ),
                "comparison_rate_percent":
                    float(
                        low_values.mean()
                        * 100
                    ),
                **low_bootstrap,
            }
        )

        low_differences = (
            count_answer_differences(
                merged=high_low,
                comparison_name="low",
            )
        )

        answer_difference_rows.append(
            {
                "region": region,
                "comparison": "low",
                "n": len(high_low),
                **low_differences,
            }
        )

        for _, row in high_random.iterrows():
            sample_rows.append(
                {
                    "sample_id":
                        row["sample_id"],
                    "region":
                        region,
                    "comparison":
                        "random",
                    "high_text":
                        row["text_high"],
                    "comparison_text":
                        row["text_random"],
                    "high_answer_class":
                        row[
                            "answer_class_high"
                        ],
                    "comparison_answer_class":
                        row[
                            "answer_class_random"
                        ],
                    "high_hallucinated":
                        row[
                            "hallucinated_high"
                        ],
                    "comparison_hallucinated":
                        row[
                            "hallucinated_random"
                        ],
                    "paired_advantage":
                        row[
                            "hallucinated_random"
                        ]
                        - row[
                            "hallucinated_high"
                        ],
                }
            )

        for _, row in high_low.iterrows():
            sample_rows.append(
                {
                    "sample_id":
                        row["sample_id"],
                    "region":
                        region,
                    "comparison":
                        "low",
                    "high_text":
                        row["text_high"],
                    "comparison_text":
                        row["text_low"],
                    "high_answer_class":
                        row[
                            "answer_class_high"
                        ],
                    "comparison_answer_class":
                        row[
                            "answer_class_low"
                        ],
                    "high_hallucinated":
                        row[
                            "hallucinated_high"
                        ],
                    "comparison_hallucinated":
                        row[
                            "hallucinated_low"
                        ],
                    "paired_advantage":
                        row[
                            "hallucinated_low"
                        ]
                        - row[
                            "hallucinated_high"
                        ],
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    paired_df = pd.DataFrame(
        paired_rows
    )

    answer_differences_df = pd.DataFrame(
        answer_difference_rows
    )

    sample_df = pd.DataFrame(
        sample_rows
    )

    summary_path = (
        METRICS_DIR
        / "qwen_low_uncertainty_control_summary.csv"
    )

    paired_path = (
        METRICS_DIR
        / "qwen_low_uncertainty_control_paired_bootstrap.csv"
    )

    answer_differences_path = (
        METRICS_DIR
        / "qwen_low_uncertainty_control_answer_flips.csv"
    )

    sample_path = (
        METRICS_DIR
        / "qwen_low_uncertainty_control_sample_level.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    paired_df.to_csv(
        paired_path,
        index=False,
    )

    answer_differences_df.to_csv(
        answer_differences_path,
        index=False,
    )

    sample_df.to_csv(
        sample_path,
        index=False,
    )

    print(
        "\nHigh vs random vs low summary"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nPaired comparisons"
    )

    print(
        paired_df.to_string(
            index=False
        )
    )

    print(
        "\nAnswer differences"
    )

    print(
        answer_differences_df.to_string(
            index=False
        )
    )

    print("\nSaved:")

    for path in [
        summary_path,
        paired_path,
        answer_differences_path,
        sample_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()