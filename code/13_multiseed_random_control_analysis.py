import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_ROOT / "metrics" / "llava7b"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

UNCERTAINTY_CONDITIONS = [
    "all",
    "removed",
    "context",
    "background",
]

RANDOM_SEEDS = [42, 43, 44, 45, 46]


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
        raise FileNotFoundError(f"Missing file: {path}")

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
            f"{path} is missing columns: {sorted(missing)}"
        )

    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")

    frame["answer_class"] = frame["text"].map(classify_answer)
    frame["hallucinated"] = (
        frame["answer_class"] == "yes"
    ).astype(int)

    return frame


def random_folder(condition: str, seed: int) -> str:
    if seed == 42:
        return f"eval_removed_random_{condition}"

    return f"eval_removed_random_{condition}_seed{seed}"


def validate_pair(
    uncertainty: pd.DataFrame,
    random_control: pd.DataFrame,
    condition: str,
    seed: int,
) -> pd.DataFrame:
    merged = uncertainty.merge(
        random_control,
        on="sample_id",
        how="inner",
        suffixes=("_uncertainty", "_random"),
        validate="one_to_one",
    )

    if len(merged) != 522:
        raise ValueError(
            f"{condition}, seed {seed}: expected 522 pairs, "
            f"found {len(merged)}"
        )

    for column in [
        "question_id",
        "label",
        "target_object",
        "prompt",
    ]:
        left = merged[f"{column}_uncertainty"]
        right = merged[f"{column}_random"]

        if not left.equals(right):
            raise ValueError(
                f"{condition}, seed {seed}: metadata mismatch "
                f"in {column}"
            )

    return merged


def paired_bootstrap(
    uncertainty_values: np.ndarray,
    random_values: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    paired_difference = random_values - uncertainty_values
    observed = float(paired_difference.mean())

    rng = np.random.default_rng(seed)
    n = len(paired_difference)

    effects = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        sampled_indices = rng.integers(0, n, size=n)
        effects[index] = paired_difference[sampled_indices].mean()

    lower, upper = np.percentile(effects, [2.5, 97.5])

    p_non_positive = np.mean(effects <= 0)
    p_non_negative = np.mean(effects >= 0)
    p_value = min(
        1.0,
        2.0 * min(p_non_positive, p_non_negative),
    )

    return {
        "advantage": observed,
        "advantage_pp": observed * 100,
        "ci_lower_pp": float(lower * 100),
        "ci_upper_pp": float(upper * 100),
        "p_value": float(p_value),
    }


def main() -> None:
    uncertainty_frames = {
        condition: load_condition(
            f"eval_removed_{condition}"
        )
        for condition in UNCERTAINTY_CONDITIONS
    }

    seed_rows = []
    sample_rows = []

    for condition in UNCERTAINTY_CONDITIONS:
        uncertainty = uncertainty_frames[condition]

        for seed in RANDOM_SEEDS:
            folder = random_folder(condition, seed)
            random_control = load_condition(folder)

            merged = validate_pair(
                uncertainty,
                random_control,
                condition,
                seed,
            )

            uncertainty_values = merged[
                "hallucinated_uncertainty"
            ].to_numpy(dtype=np.int64)

            random_values = merged[
                "hallucinated_random"
            ].to_numpy(dtype=np.int64)

            uncertainty_rate = float(
                uncertainty_values.mean()
            )
            random_rate = float(
                random_values.mean()
            )

            bootstrap = paired_bootstrap(
                uncertainty_values,
                random_values,
                seed=1000 + seed,
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

            seed_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "n": len(merged),
                    "uncertainty_hallucination_rate":
                        uncertainty_rate,
                    "random_hallucination_rate":
                        random_rate,
                    "uncertainty_hallucination_rate_percent":
                        uncertainty_rate * 100,
                    "random_hallucination_rate_percent":
                        random_rate * 100,
                    "uncertainty_advantage_pp":
                        bootstrap["advantage_pp"],
                    "ci_lower_pp":
                        bootstrap["ci_lower_pp"],
                    "ci_upper_pp":
                        bootstrap["ci_upper_pp"],
                    "p_value":
                        bootstrap["p_value"],
                    "uncertainty_yes_random_no":
                        uncertainty_yes_random_no,
                    "uncertainty_no_random_yes":
                        uncertainty_no_random_yes,
                    "net_uncertainty_advantage":
                        (
                            uncertainty_no_random_yes
                            - uncertainty_yes_random_no
                        ),
                }
            )

            for _, row in merged.iterrows():
                sample_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "sample_id": row["sample_id"],
                        "target_object":
                            row["target_object_uncertainty"],
                        "uncertainty_answer":
                            row["answer_class_uncertainty"],
                        "random_answer":
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

    seed_df = pd.DataFrame(seed_rows)
    sample_df = pd.DataFrame(sample_rows)

    aggregate_rows = []

    for condition, group in seed_df.groupby("condition"):
        uncertainty_rate = float(
            group[
                "uncertainty_hallucination_rate_percent"
            ].iloc[0]
        )

        random_mean = float(
            group[
                "random_hallucination_rate_percent"
            ].mean()
        )

        random_std = float(
            group[
                "random_hallucination_rate_percent"
            ].std(ddof=1)
        )

        advantage_mean = float(
            group["uncertainty_advantage_pp"].mean()
        )

        advantage_std = float(
            group["uncertainty_advantage_pp"].std(ddof=1)
        )

        aggregate_rows.append(
            {
                "condition": condition,
                "num_random_seeds": len(group),
                "uncertainty_hallucination_rate_percent":
                    uncertainty_rate,
                "random_hallucination_rate_mean_percent":
                    random_mean,
                "random_hallucination_rate_std_percent":
                    random_std,
                "uncertainty_advantage_mean_pp":
                    advantage_mean,
                "uncertainty_advantage_std_pp":
                    advantage_std,
                "minimum_random_rate_percent":
                    float(
                        group[
                            "random_hallucination_rate_percent"
                        ].min()
                    ),
                "maximum_random_rate_percent":
                    float(
                        group[
                            "random_hallucination_rate_percent"
                        ].max()
                    ),
                "seeds_where_uncertainty_better":
                    int(
                        (
                            group["uncertainty_advantage_pp"] > 0
                        ).sum()
                    ),
                "seeds_where_random_better":
                    int(
                        (
                            group["uncertainty_advantage_pp"] < 0
                        ).sum()
                    ),
                "seeds_tied":
                    int(
                        (
                            group["uncertainty_advantage_pp"] == 0
                        ).sum()
                    ),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows)

    seed_path = (
        METRICS_DIR
        / "random_control_multiseed_per_seed.csv"
    )

    aggregate_path = (
        METRICS_DIR
        / "random_control_multiseed_summary.csv"
    )

    sample_path = (
        METRICS_DIR
        / "random_control_multiseed_sample_level.csv"
    )

    seed_df.to_csv(seed_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    sample_df.to_csv(sample_path, index=False)

    print("\nPer-seed results")
    print(
        seed_df[
            [
                "condition",
                "seed",
                "uncertainty_hallucination_rate_percent",
                "random_hallucination_rate_percent",
                "uncertainty_advantage_pp",
                "ci_lower_pp",
                "ci_upper_pp",
                "p_value",
            ]
        ].to_string(index=False)
    )

    print("\nMulti-seed summary")
    print(aggregate_df.to_string(index=False))

    print("\nSaved:")
    print(seed_path)
    print(aggregate_path)
    print(sample_path)


if __name__ == "__main__":
    main()