import pandas as pd
from pathlib import Path

ROOT = Path("outputs") / "metrics"

MODELS = {
    "llava7b": ROOT / "llava7b",
    "llava13b": ROOT / "llava13b",
}

MODES = ["all", "removed", "context", "background"]


def to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ["true", "1", "yes"]


def load_model(folder):
    captions_path = folder / "removed_eval_all_captions_long.csv"
    region_path = folder / "removed_eval_region_uncertainty_long.csv"

    if not captions_path.exists():
        raise FileNotFoundError(f"Missing: {captions_path}")

    if not region_path.exists():
        raise FileNotFoundError(f"Missing: {region_path}")

    captions = pd.read_csv(captions_path)
    region = pd.read_csv(region_path)

    print()
    print("Loaded:", folder)
    print("captions columns:", list(captions.columns))
    print("region columns:", list(region.columns))

    captions["hallucinated"] = captions["hallucinated"].apply(to_bool)

    return captions, region


def get_active_samples(region_df, mode):
    sub = region_df[region_df["condition"] == mode].copy()

    if sub.empty:
        raise RuntimeError(f"No rows found for condition={mode}")

    if mode == "all":
        # For all-token masking, active if total suppressed tokens > 0.
        active = (
            sub.groupby("sample_id")["num_suppressed_patch_tokens"]
            .max()
            .reset_index()
        )
        active = active[active["num_suppressed_patch_tokens"] > 0]
        return set(active["sample_id"])

    # For region-specific masking, active if that specific region had suppressed tokens.
    region_sub = sub[sub["region"] == mode].copy()

    if region_sub.empty:
        raise RuntimeError(f"No rows found for condition={mode}, region={mode}")

    active = region_sub[region_sub["region_suppressed"] > 0]
    return set(active["sample_id"])


def analyze_model(model_name, folder):
    captions, region = load_model(folder)

    rows = []

    print()
    print(f"===== {model_name}: active suppression ablation =====")
    print(
        "condition,total_samples,active_samples,active_fraction,"
        "none_hallucinated,masked_hallucinated,"
        "none_rate,masked_rate,effect,effect_pp,"
        "yes_to_no,no_to_yes,net_hallucination_reduction"
    )

    none = captions[captions["condition"] == "none"].copy()

    if none.empty:
        raise RuntimeError(f"No condition='none' rows found for {model_name}")

    for mode in MODES:
        masked = captions[captions["condition"] == mode].copy()

        if masked.empty:
            raise RuntimeError(f"No condition='{mode}' rows found for {model_name}")

        active_samples = get_active_samples(region, mode)

        none_active = none[none["sample_id"].isin(active_samples)].copy()
        masked_active = masked[masked["sample_id"].isin(active_samples)].copy()

        # Align by sample_id
        merged = none_active[
            ["sample_id", "hallucinated", "answer_class"]
        ].merge(
            masked_active[["sample_id", "hallucinated", "answer_class"]],
            on="sample_id",
            suffixes=("_none", "_masked"),
        )

        total_samples = len(none)
        active_n = len(merged)

        none_h = int(merged["hallucinated_none"].sum())
        masked_h = int(merged["hallucinated_masked"].sum())

        none_rate = none_h / active_n if active_n else 0.0
        masked_rate = masked_h / active_n if active_n else 0.0
        effect = none_rate - masked_rate

        yes_to_no = int(
            (
                (merged["answer_class_none"] == "yes")
                & (merged["answer_class_masked"] == "no")
            ).sum()
        )

        no_to_yes = int(
            (
                (merged["answer_class_none"] == "no")
                & (merged["answer_class_masked"] == "yes")
            ).sum()
        )

        net = yes_to_no - no_to_yes

        row = {
            "model": model_name,
            "condition": mode,
            "total_samples": total_samples,
            "active_samples": active_n,
            "active_fraction": active_n / total_samples,
            "none_hallucinated": none_h,
            "masked_hallucinated": masked_h,
            "none_rate": none_rate,
            "masked_rate": masked_rate,
            "effect": effect,
            "effect_pp": effect * 100,
            "yes_to_no": yes_to_no,
            "no_to_yes": no_to_yes,
            "net_hallucination_reduction": net,
        }

        rows.append(row)

        print(
            f"{mode},{total_samples},{active_n},{active_n/total_samples:.6f},"
            f"{none_h},{masked_h},{none_rate:.6f},{masked_rate:.6f},"
            f"{effect:.6f},{effect*100:.3f},"
            f"{yes_to_no},{no_to_yes},{net}"
        )

    out_path = folder / "removed_eval_active_suppression_ablation.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)

    return rows


all_rows = []

for model_name, folder in MODELS.items():
    rows = analyze_model(model_name, folder)
    all_rows.extend(rows)

comparison = pd.DataFrame(all_rows)
comparison_path = ROOT / "active_suppression_ablation_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print()
print("Saved combined comparison:", comparison_path)