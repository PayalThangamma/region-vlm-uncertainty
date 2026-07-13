from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics" / "llava7b"

INPUT_PATH = (
    METRICS_DIR
    / "random_control_multiseed_sample_level.csv"
)

OUTPUT_PATH = (
    METRICS_DIR
    / "random_control_multiseed_pooled_bootstrap.csv"
)

N_BOOTSTRAP = 10_000
SEED = 42


def bootstrap_condition(
    frame: pd.DataFrame,
    seed: int,
) -> dict:
    # One uncertainty result per sample, repeated across seeds.
    uncertainty = (
        frame.groupby("sample_id")["uncertainty_hallucinated"]
        .first()
        .sort_index()
    )

    # Mean random hallucination indicator across seeds 42–46.
    random_mean = (
        frame.groupby("sample_id")["random_hallucinated"]
        .mean()
        .sort_index()
    )

    if not uncertainty.index.equals(random_mean.index):
        raise ValueError("Sample IDs do not align.")

    paired_advantage = (
        random_mean.to_numpy(dtype=float)
        - uncertainty.to_numpy(dtype=float)
    )

    observed = float(paired_advantage.mean())

    rng = np.random.default_rng(seed)
    n = len(paired_advantage)

    bootstrap_effects = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    for index in range(N_BOOTSTRAP):
        sampled = rng.integers(0, n, size=n)
        bootstrap_effects[index] = paired_advantage[
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
        "n_samples": n,
        "uncertainty_rate_percent":
            float(uncertainty.mean() * 100),
        "random_mean_rate_percent":
            float(random_mean.mean() * 100),
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
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    data = pd.read_csv(INPUT_PATH)

    required = {
        "condition",
        "seed",
        "sample_id",
        "uncertainty_hallucinated",
        "random_hallucinated",
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    rows = []

    for index, condition in enumerate(
        ["all", "removed", "context", "background"]
    ):
        subset = data[
            data["condition"] == condition
        ].copy()

        result = bootstrap_condition(
            subset,
            seed=SEED + index,
        )

        rows.append(
            {
                "condition": condition,
                "num_random_seeds":
                    subset["seed"].nunique(),
                **result,
            }
        )

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\nPooled multi-seed bootstrap")
    print(output.to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()