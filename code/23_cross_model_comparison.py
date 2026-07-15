from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

OUTPUT_DIR = (
    METRICS_ROOT
    / "cross_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODELS = {
    "LLaVA-1.5-7B": {
        "folder": METRICS_ROOT / "llava7b",
        "removed_summary":
            "removed_eval_summary.csv",
        "removed_bootstrap":
            "removed_eval_bootstrap_effects.csv",
        "original_summary":
            "original_sanity_summary.csv",
        "original_bootstrap":
            "original_sanity_paired_bootstrap.csv",
    },
    "LLaVA-1.5-13B": {
        "folder": METRICS_ROOT / "llava13b",
        "removed_summary":
            "removed_eval_summary.csv",
        "removed_bootstrap":
            "removed_eval_bootstrap_effects.csv",
        "original_summary":
            None,
        "original_bootstrap":
            None,
    },
    "Qwen2.5-VL-7B": {
        "folder": METRICS_ROOT / "qwen25vl7b",
        "removed_summary":
            "qwen_removed_summary.csv",
        "removed_bootstrap":
            "qwen_removed_bootstrap_effects.csv",
        "original_summary":
            "qwen_original_sanity_summary.csv",
        "original_bootstrap":
            "qwen_original_sanity_paired_bootstrap.csv",
    },
}


CONDITION_ORDER = [
    "none",
    "all",
    "removed",
    "context",
    "background",
]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    return path


def normalize_removed_summary(
    model_name: str,
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        require_file(path)
    )

    required = {
        "condition",
        "n",
        "hallucinated_yes",
        "correct_rejection_no",
        "hallucination_rate",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    if "unknown" not in frame.columns:
        frame["unknown"] = 0

    if (
        "hallucination_rate_percent"
        not in frame.columns
    ):
        frame[
            "hallucination_rate_percent"
        ] = (
            frame["hallucination_rate"]
            * 100
        )

    frame["model"] = model_name

    return frame[
        [
            "model",
            "condition",
            "n",
            "hallucinated_yes",
            "correct_rejection_no",
            "unknown",
            "hallucination_rate",
            "hallucination_rate_percent",
        ]
    ]


def normalize_removed_bootstrap(
    model_name: str,
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_csv(
        require_file(path)
    )

    if "causal_effect_pp" not in frame.columns:

        if "observed_effect_percentage_points" in frame.columns:
            frame = frame.rename(
                columns={
                    "observed_effect_percentage_points":
                        "causal_effect_pp"
                }
            )

        elif "effect_pp" in frame.columns:
            frame = frame.rename(
                columns={
                    "effect_pp":
                        "causal_effect_pp"
                }
            )

        elif "causal_effect" in frame.columns:
            frame["causal_effect_pp"] = (
                frame["causal_effect"] * 100
            )

        elif "observed_effect" in frame.columns:
            frame["causal_effect_pp"] = (
                frame["observed_effect"] * 100
            )

        else:
            raise ValueError(
                f"{path} does not contain an effect column."
            )

    if "ci_lower_pp" not in frame.columns:

        if "ci_low_percentage_points" in frame.columns:
            frame = frame.rename(
                columns={
                    "ci_low_percentage_points":
                        "ci_lower_pp"
                }
            )

        elif "ci_low" in frame.columns:
            frame["ci_lower_pp"] = (
                frame["ci_low"] * 100
            )

    if "ci_upper_pp" not in frame.columns:

        if "ci_high_percentage_points" in frame.columns:
            frame = frame.rename(
                columns={
                    "ci_high_percentage_points":
                        "ci_upper_pp"
                }
            )

        elif "ci_high" in frame.columns:
            frame["ci_upper_pp"] = (
                frame["ci_high"] * 100
            )

    if "p_value" not in frame.columns:

        if "p_value_approx" in frame.columns:
            frame = frame.rename(
                columns={
                    "p_value_approx":
                        "p_value"
                }
            )

    if "baseline_rate_percent" not in frame.columns:

        if "none_rate" in frame.columns:
            frame["baseline_rate_percent"] = (
                frame["none_rate"] * 100
            )
        else:
            frame["baseline_rate_percent"] = pd.NA

    if "masked_rate_percent" not in frame.columns:

        if "masked_rate" in frame.columns:
            frame["masked_rate_percent"] = (
                frame["masked_rate"] * 100
            )
        else:
            frame["masked_rate_percent"] = pd.NA

    required = {
        "condition",
        "causal_effect_pp",
        "ci_lower_pp",
        "ci_upper_pp",
        "p_value",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{path} missing columns: {sorted(missing)}"
        )

    frame["model"] = model_name

    return frame[
        [
            "model",
            "condition",
            "baseline_rate_percent",
            "masked_rate_percent",
            "causal_effect_pp",
            "ci_lower_pp",
            "ci_upper_pp",
            "p_value",
        ]
    ]

def normalize_original_summary(
    model_name: str,
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        require_file(path)
    )

    required = {
        "condition",
        "n",
        "correct_yes",
        "false_negative_no",
        "accuracy",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    if "unknown" not in frame.columns:
        frame["unknown"] = 0

    if "accuracy_percent" not in frame.columns:
        frame["accuracy_percent"] = (
            frame["accuracy"]
            * 100
        )

    if (
        "false_negative_rate_percent"
        not in frame.columns
    ):
        frame[
            "false_negative_rate_percent"
        ] = (
            frame["false_negative_no"]
            / frame["n"]
            * 100
        )

    frame["model"] = model_name

    return frame[
        [
            "model",
            "condition",
            "n",
            "correct_yes",
            "false_negative_no",
            "unknown",
            "accuracy",
            "accuracy_percent",
            "false_negative_rate_percent",
        ]
    ]


def normalize_original_bootstrap(
    model_name: str,
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        require_file(path)
    )

    required = {
        "condition",
        "baseline_accuracy_percent",
        "masked_accuracy_percent",
        "accuracy_drop_pp",
        "ci_lower_pp",
        "ci_upper_pp",
        "p_value",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    frame["model"] = model_name

    return frame[
        [
            "model",
            "condition",
            "baseline_accuracy_percent",
            "masked_accuracy_percent",
            "accuracy_drop_pp",
            "ci_lower_pp",
            "ci_upper_pp",
            "p_value",
        ]
    ]


def main() -> None:
    removed_summaries = []
    removed_bootstraps = []
    original_summaries = []
    original_bootstraps = []

    for model_name, config in MODELS.items():
        folder = config["folder"]

        removed_summaries.append(
            normalize_removed_summary(
                model_name,
                folder
                / config[
                    "removed_summary"
                ],
            )
        )

        removed_bootstraps.append(
            normalize_removed_bootstrap(
                model_name,
                folder
                / config[
                    "removed_bootstrap"
                ],
            )
        )

        original_summary_name = (
            config["original_summary"]
        )

        original_bootstrap_name = (
            config["original_bootstrap"]
        )

        if original_summary_name is not None:
            original_summaries.append(
                normalize_original_summary(
                    model_name,
                    folder
                    / original_summary_name,
                )
            )

        if original_bootstrap_name is not None:
            original_bootstraps.append(
                normalize_original_bootstrap(
                    model_name,
                    folder
                    / original_bootstrap_name,
                )
            )

    removed_summary_df = pd.concat(
        removed_summaries,
        ignore_index=True,
    )

    removed_bootstrap_df = pd.concat(
        removed_bootstraps,
        ignore_index=True,
    )

    original_summary_df = pd.concat(
        original_summaries,
        ignore_index=True,
    )

    original_bootstrap_df = pd.concat(
        original_bootstraps,
        ignore_index=True,
    )

    condition_rank = {
        condition: index
        for index, condition
        in enumerate(CONDITION_ORDER)
    }

    removed_summary_df[
        "condition_rank"
    ] = removed_summary_df[
        "condition"
    ].map(condition_rank)

    removed_summary_df = (
        removed_summary_df
        .sort_values(
            [
                "model",
                "condition_rank",
            ]
        )
        .drop(
            columns="condition_rank"
        )
    )

    removed_bootstrap_df[
        "condition_rank"
    ] = removed_bootstrap_df[
        "condition"
    ].map(condition_rank)

    removed_bootstrap_df = (
        removed_bootstrap_df
        .sort_values(
            [
                "model",
                "condition_rank",
            ]
        )
        .drop(
            columns="condition_rank"
        )
    )

    original_summary_df[
        "condition_rank"
    ] = original_summary_df[
        "condition"
    ].map(condition_rank)

    original_summary_df = (
        original_summary_df
        .sort_values(
            [
                "model",
                "condition_rank",
            ]
        )
        .drop(
            columns="condition_rank"
        )
    )

    original_bootstrap_df[
        "condition_rank"
    ] = original_bootstrap_df[
        "condition"
    ].map(condition_rank)

    original_bootstrap_df = (
        original_bootstrap_df
        .sort_values(
            [
                "model",
                "condition_rank",
            ]
        )
        .drop(
            columns="condition_rank"
        )
    )

    effect_table = (
        removed_bootstrap_df
        .pivot(
            index="model",
            columns="condition",
            values="causal_effect_pp",
        )
        .reset_index()
    )

    for condition in [
        "all",
        "removed",
        "context",
        "background",
    ]:
        if condition not in effect_table.columns:
            effect_table[condition] = pd.NA

    effect_table = effect_table[
        [
            "model",
            "all",
            "removed",
            "context",
            "background",
        ]
    ]

    baseline_table = (
        removed_summary_df[
            removed_summary_df[
                "condition"
            ] == "none"
        ][
            [
                "model",
                "hallucination_rate_percent",
            ]
        ]
        .rename(
            columns={
                "hallucination_rate_percent":
                    "baseline_hallucination_rate_percent"
            }
        )
    )

    sanity_effect_table = (
        original_bootstrap_df
        .pivot(
            index="model",
            columns="condition",
            values="accuracy_drop_pp",
        )
        .reset_index()
    )

    final_overview = baseline_table.merge(
        effect_table,
        on="model",
        how="left",
        validate="one_to_one",
    )

    final_overview = final_overview.rename(
        columns={
            "all":
                "all_hallucination_reduction_pp",
            "removed":
                "removed_hallucination_reduction_pp",
            "context":
                "context_hallucination_reduction_pp",
            "background":
                "background_hallucination_reduction_pp",
        }
    )

    if not sanity_effect_table.empty:
        sanity_effect_table = (
            sanity_effect_table.rename(
                columns={
                    "all":
                        "all_accuracy_drop_pp",
                    "removed":
                        "removed_accuracy_drop_pp",
                    "context":
                        "context_accuracy_drop_pp",
                    "background":
                        "background_accuracy_drop_pp",
                }
            )
        )

        final_overview = (
            final_overview.merge(
                sanity_effect_table,
                on="model",
                how="left",
            )
        )

    files = {
        "cross_model_removed_summary.csv":
            removed_summary_df,
        "cross_model_removed_bootstrap.csv":
            removed_bootstrap_df,
        "cross_model_original_summary.csv":
            original_summary_df,
        "cross_model_original_bootstrap.csv":
            original_bootstrap_df,
        "cross_model_effect_table.csv":
            effect_table,
        "cross_model_final_overview.csv":
            final_overview,
    }

    for filename, frame in files.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(
            path,
            index=False,
        )

    print(
        "\nCross-model causal effects "
        "(positive = hallucination reduction)"
    )

    print(
        effect_table.to_string(
            index=False
        )
    )

    print(
        "\nCross-model final overview"
    )

    print(
        final_overview.to_string(
            index=False
        )
    )

    print("\nSaved:")

    for filename in files:
        print(
            OUTPUT_DIR / filename
        )


if __name__ == "__main__":
    main()