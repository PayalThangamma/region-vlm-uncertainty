from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

CROSS_MODEL_ROOT = (
    METRICS_ROOT
    / "cross_model"
)

QWEN_ROOT = (
    METRICS_ROOT
    / "qwen25vl7b"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "plots"
    / "final"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_ORDER = [
    "LLaVA-1.5-7B",
    "LLaVA-1.5-13B",
    "Qwen2.5-VL-7B",
]

CONDITION_ORDER = [
    "all",
    "removed",
    "context",
    "background",
]

CONDITION_LABELS = {
    "all": "All",
    "removed": "Removed",
    "context": "Context",
    "background": "Background",
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    return path


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    output_path = PLOT_DIR / filename

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", output_path)


def add_bar_labels(
    axis: plt.Axes,
    bars,
    decimals: int = 2,
) -> None:
    for bar in bars:
        value = bar.get_height()

        vertical_alignment = (
            "bottom"
            if value >= 0
            else "top"
        )

        offset = (
            3
            if value >= 0
            else -3
        )

        axis.annotate(
            f"{value:.{decimals}f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                value,
            ),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=8,
        )


def plot_baseline_hallucination(
    final_overview: pd.DataFrame,
) -> None:
    frame = (
        final_overview
        .set_index("model")
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    bars = axis.bar(
        frame["model"],
        frame[
            "baseline_hallucination_rate_percent"
        ],
    )

    add_bar_labels(axis, bars)

    axis.set_title(
        "Baseline hallucination rate by model"
    )

    axis.set_ylabel(
        "Hallucination rate (%)"
    )

    axis.set_ylim(
        0,
        max(
            100,
            frame[
                "baseline_hallucination_rate_percent"
            ].max() + 10,
        ),
    )

    axis.tick_params(
        axis="x",
        rotation=15,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    save_figure(
        figure,
        "baseline_hallucination_rate.png",
    )


def plot_causal_effects(
    bootstrap: pd.DataFrame,
) -> None:
    frame = bootstrap.copy()

    frame["model"] = pd.Categorical(
        frame["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    frame["condition"] = pd.Categorical(
        frame["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )

    frame = frame.sort_values(
        ["model", "condition"]
    )

    x_positions = np.arange(
        len(CONDITION_ORDER)
    )

    width = 0.24

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for model_index, model in enumerate(
        MODEL_ORDER
    ):
        model_frame = (
            frame[
                frame["model"] == model
            ]
            .set_index("condition")
            .reindex(CONDITION_ORDER)
        )

        values = model_frame[
            "causal_effect_pp"
        ].to_numpy(dtype=float)

        lower = model_frame[
            "ci_lower_pp"
        ].to_numpy(dtype=float)

        upper = model_frame[
            "ci_upper_pp"
        ].to_numpy(dtype=float)

        lower_error = values - lower
        upper_error = upper - values

        positions = (
            x_positions
            + (
                model_index
                - (len(MODEL_ORDER) - 1) / 2
            )
            * width
        )

        axis.bar(
            positions,
            values,
            width=width,
            label=model,
            yerr=np.vstack(
                [
                    lower_error,
                    upper_error,
                ]
            ),
            capsize=3,
        )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.set_title(
        "Hallucination reduction from visual-token masking"
    )

    axis.set_ylabel(
        "Causal effect (percentage points)"
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [
            CONDITION_LABELS[condition]
            for condition in CONDITION_ORDER
        ]
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    save_figure(
        figure,
        "cross_model_causal_effects.png",
    )


def plot_original_accuracy_drop(
    original_bootstrap: pd.DataFrame,
) -> None:
    frame = original_bootstrap.copy()

    frame = frame[
        frame["model"].isin(
            [
                "LLaVA-1.5-7B",
                "Qwen2.5-VL-7B",
            ]
        )
    ]

    models = [
        "LLaVA-1.5-7B",
        "Qwen2.5-VL-7B",
    ]

    x_positions = np.arange(
        len(CONDITION_ORDER)
    )

    width = 0.34

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for model_index, model in enumerate(
        models
    ):
        model_frame = (
            frame[
                frame["model"] == model
            ]
            .set_index("condition")
            .reindex(CONDITION_ORDER)
        )

        values = model_frame[
            "accuracy_drop_pp"
        ].to_numpy(dtype=float)

        lower = model_frame[
            "ci_lower_pp"
        ].to_numpy(dtype=float)

        upper = model_frame[
            "ci_upper_pp"
        ].to_numpy(dtype=float)

        lower_error = values - lower
        upper_error = upper - values

        positions = (
            x_positions
            + (
                model_index
                - (len(models) - 1) / 2
            )
            * width
        )

        axis.bar(
            positions,
            values,
            width=width,
            label=model,
            yerr=np.vstack(
                [
                    lower_error,
                    upper_error,
                ]
            ),
            capsize=3,
        )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.set_title(
        "Original-image accuracy drop after masking"
    )

    axis.set_ylabel(
        "Accuracy drop (percentage points)"
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [
            CONDITION_LABELS[condition]
            for condition in CONDITION_ORDER
        ]
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    save_figure(
        figure,
        "original_image_accuracy_drop.png",
    )


def plot_qwen_controls(
    qwen_control_summary: pd.DataFrame,
) -> None:
    frame = qwen_control_summary.copy()

    strategy_order = [
        "high_uncertainty",
        "matched_random_seed42",
        "matched_low_uncertainty",
    ]

    strategy_labels = {
        "high_uncertainty":
            "High uncertainty",
        "matched_random_seed42":
            "Random matched",
        "matched_low_uncertainty":
            "Low uncertainty",
    }

    x_positions = np.arange(
        len(CONDITION_ORDER)
    )

    width = 0.24

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for strategy_index, strategy in enumerate(
        strategy_order
    ):
        strategy_frame = (
            frame[
                frame["strategy"] == strategy
            ]
            .set_index("region")
            .reindex(CONDITION_ORDER)
        )

        values = strategy_frame[
            "hallucination_rate_percent"
        ].to_numpy(dtype=float)

        positions = (
            x_positions
            + (
                strategy_index
                - (len(strategy_order) - 1) / 2
            )
            * width
        )

        axis.bar(
            positions,
            values,
            width=width,
            label=strategy_labels[strategy],
        )

    axis.set_title(
        "Qwen high-, random-, and low-uncertainty controls"
    )

    axis.set_ylabel(
        "Hallucination rate (%)"
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [
            CONDITION_LABELS[condition]
            for condition in CONDITION_ORDER
        ]
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    save_figure(
        figure,
        "qwen_uncertainty_controls.png",
    )


def plot_qwen_answer_flips(
    answer_flips: pd.DataFrame,
) -> None:
    frame = (
        answer_flips
        .set_index("condition")
        .reindex(CONDITION_ORDER)
        .reset_index()
    )

    x_positions = np.arange(
        len(CONDITION_ORDER)
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.bar(
        x_positions - width / 2,
        frame["yes_to_no"],
        width=width,
        label="Yes to No",
    )

    axis.bar(
        x_positions + width / 2,
        frame["no_to_yes"],
        width=width,
        label="No to Yes",
    )

    axis.set_title(
        "Qwen answer flips relative to no masking"
    )

    axis.set_ylabel(
        "Number of samples"
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [
            CONDITION_LABELS[condition]
            for condition in CONDITION_ORDER
        ]
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    save_figure(
        figure,
        "qwen_answer_flips.png",
    )


def create_final_table(
    final_overview: pd.DataFrame,
) -> None:
    frame = (
        final_overview
        .set_index("model")
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    selected = frame[
        [
            "model",
            "baseline_hallucination_rate_percent",
            "all_hallucination_reduction_pp",
            "removed_hallucination_reduction_pp",
            "context_hallucination_reduction_pp",
            "background_hallucination_reduction_pp",
            "all_accuracy_drop_pp",
            "background_accuracy_drop_pp",
            "removed_accuracy_drop_pp",
            "context_accuracy_drop_pp",
        ]
    ].copy()

    numeric_columns = selected.columns[
        selected.columns != "model"
    ]

    selected[numeric_columns] = (
        selected[numeric_columns]
        .round(2)
    )

    output_path = (
        CROSS_MODEL_ROOT
        / "cross_model_summary.csv"
    )

    selected.to_csv(
        output_path,
        index=False,
    )

    print("\nSummary")
    print(
        selected.to_string(
            index=False
        )
    )

    print("Saved:", output_path)


def main() -> None:
    final_overview = pd.read_csv(
        require_file(
            CROSS_MODEL_ROOT
            / "cross_model_final_overview.csv"
        )
    )

    removed_bootstrap = pd.read_csv(
        require_file(
            CROSS_MODEL_ROOT
            / "cross_model_removed_bootstrap.csv"
        )
    )

    original_bootstrap = pd.read_csv(
        require_file(
            CROSS_MODEL_ROOT
            / "cross_model_original_bootstrap.csv"
        )
    )

    qwen_control_summary = pd.read_csv(
        require_file(
            QWEN_ROOT
            / "qwen_low_uncertainty_control_summary.csv"
        )
    )

    qwen_answer_flips = pd.read_csv(
        require_file(
            QWEN_ROOT
            / "qwen_removed_answer_flips.csv"
        )
    )

    plot_baseline_hallucination(
        final_overview
    )

    plot_causal_effects(
        removed_bootstrap
    )

    plot_original_accuracy_drop(
        original_bootstrap
    )

    plot_qwen_controls(
        qwen_control_summary
    )

    plot_qwen_answer_flips(
        qwen_answer_flips
    )

    create_final_table(
        final_overview
    )

    print(
        "\nAll final cross-model plots created."
    )


if __name__ == "__main__":
    main()