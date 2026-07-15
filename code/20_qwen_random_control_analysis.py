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
SEED = 42


def classify_answer(text: str) -> str:
    normalized = text.strip().lower()

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

    if len(frame) != 522:
        raise ValueError(
            f"{path}: expected 522 rows, found {len(frame)}"
        )

    if frame["sample_id"].duplicated().any():
        raise ValueError(
            f"{path}: duplicate sample IDs found"
        )

    frame["answer_class"] = frame["text"].map(
        classify_answer
    )

    frame["hallucinated"] = (
        frame["answer_class"] == "yes"
    ).astype(int)

    return frame


def paired_bootstrap(
    high_values: np.ndarray,
    random_values: np.ndarray,
    seed: int,
) -> dict:
    # Positive means high-uncertainty masking is better.
    paired_effect = random_values - high_values

    observed = float(paired_effect.mean())

    rng = np.random.default_rng(seed)
    n = len(paired_effect)

    bootstrap_effects = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    for index in range(N_BOOTSTRAP):
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
        "uncertainty_advantage":
            observed,
        "uncertainty_advantage_pp":
            observed * 100,
        "ci_lower_pp":
            float(lower * 100),
        "ci_upper_pp":
            float(upper * 100),
        "p_value":
            float(p_value),
    }


def main() -> None:
    summary_rows = []
    comparison_rows = []
    flip_rows = []
    sample_rows = []

    for index, region in enumerate(REGIONS):
        high = load_jsonl(
            HIGH_ROOT / f"captions_{region}.jsonl"
        )

        random = load_jsonl(
            RANDOM_ROOT
            / f"captions_random_{region}.jsonl"
        )

        merged = high.merge(
            random,
            on="sample_id",
            how="inner",
            suffixes=("_high", "_random"),
            validate="one_to_one",
        )

        if len(merged) != 522:
            raise ValueError(
                f"{region}: expected 522 paired rows"
            )

        high_values = merged[
            "hallucinated_high"
        ].to_numpy(dtype=np.int64)

        random_values = merged[
            "hallucinated_random"
        ].to_numpy(dtype=np.int64)

        high_rate = float(high_values.mean())
        random_rate = float(random_values.mean())

        summary_rows.extend(
            [
                {
                    "region": region,
                    "strategy": "high_uncertainty",
                    "n": 522,
                    "hallucination_rate":
                        high_rate,
                    "hallucination_rate_percent":
                        high_rate * 100,
                },
                {
                    "region": region,
                    "strategy": "random_matched_seed42",
                    "n": 522,
                    "hallucination_rate":
                        random_rate,
                    "hallucination_rate_percent":
                        random_rate * 100,
                },
            ]
        )

        comparison = paired_bootstrap(
            high_values,
            random_values,
            seed=SEED + index,
        )

        comparison_rows.append(
            {
                "region": region,
                "n": 522,
                "high_rate_percent":
                    high_rate * 100,
                "random_rate_percent":
                    random_rate * 100,
                **comparison,
            }
        )

        high_yes_random_no = int(
            (
                (merged["answer_class_high"] == "yes")
                & (
                    merged["answer_class_random"]
                    == "no"
                )
            ).sum()
        )

        high_no_random_yes = int(
            (
                (merged["answer_class_high"] == "no")
                & (
                    merged["answer_class_random"]
                    == "yes"
                )
            ).sum()
        )

        unchanged_yes = int(
            (
                (merged["answer_class_high"] == "yes")
                & (
                    merged["answer_class_random"]
                    == "yes"
                )
            ).sum()
        )

        unchanged_no = int(
            (
                (merged["answer_class_high"] == "no")
                & (
                    merged["answer_class_random"]
                    == "no"
                )
            ).sum()
        )

        unknown_changes = int(
            (
                (merged["answer_class_high"] == "unknown")
                | (
                    merged["answer_class_random"]
                    == "unknown"
                )
            ).sum()
        )

        flip_rows.append(
            {
                "region": region,
                "n": 522,
                "high_yes_random_no":
                    high_yes_random_no,
                "high_no_random_yes":
                    high_no_random_yes,
                "unchanged_yes":
                    unchanged_yes,
                "unchanged_no":
                    unchanged_no,
                "unknown_changes":
                    unknown_changes,
                "net_uncertainty_advantage":
                    high_no_random_yes
                    - high_yes_random_no,
            }
        )

        for _, row in merged.iterrows():
            sample_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "region": region,
                    "high_answer":
                        row["answer_class_high"],
                    "random_answer":
                        row["answer_class_random"],
                    "high_hallucinated":
                        row["hallucinated_high"],
                    "random_hallucinated":
                        row["hallucinated_random"],
                    "paired_advantage":
                        row["hallucinated_random"]
                        - row["hallucinated_high"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    flips_df = pd.DataFrame(flip_rows)
    sample_df = pd.DataFrame(sample_rows)

    summary_path = (
        METRICS_DIR
        / "qwen_random_control_seed42_summary.csv"
    )

    comparison_path = (
        METRICS_DIR
        / "qwen_random_control_seed42_bootstrap.csv"
    )

    flips_path = (
        METRICS_DIR
        / "qwen_random_control_seed42_answer_flips.csv"
    )

    sample_path = (
        METRICS_DIR
        / "qwen_random_control_seed42_sample_level.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    flips_df.to_csv(flips_path, index=False)
    sample_df.to_csv(sample_path, index=False)

    print("\nQwen matched-random summary")
    print(summary_df.to_string(index=False))

    print("\nPaired high-uncertainty vs random")
    print(comparison_df.to_string(index=False))

    print("\nAnswer differences")
    print(flips_df.to_string(index=False))

    print("\nSaved:")
    print(summary_path)
    print(comparison_path)
    print(flips_path)
    print(sample_path)


if __name__ == "__main__":
    main()