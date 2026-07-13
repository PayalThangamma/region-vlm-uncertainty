import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "none": ROOT / "outputs" /"llava13b"/ "eval_removed_none_llava13b",
    "all": ROOT / "outputs" / "llava13b"/ "eval_removed_all_llava13b",
    "removed": ROOT / "outputs" /"llava13b"/  "eval_removed_removed_llava13b",
    "context": ROOT / "outputs" / "llava13b"/ "eval_removed_context_llava13b",
    "background": ROOT / "outputs" / "llava13b"/ "eval_removed_background_llava13b",
}

METRICS_DIR = ROOT / "outputs" / "metrics"/"llava13b"
PLOTS_DIR = ROOT / "outputs" / "plots" / "llava13b"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def classify_answer(text: str) -> str:
    """
    For removed-object questions, label is no.
    If the answer starts with yes, it is hallucination-like.
    If it starts with no, it is correct-rejection-like.
    Otherwise it is unclear.
    """
    t = normalize_text(text)

    if re.match(r"^(yes|yeah|yep)\b", t):
        return "yes"
    if re.match(r"^(no|nope)\b", t):
        return "no"
    return "other"


def load_captions(run_name: str, run_dir: Path) -> pd.DataFrame:
    path = run_dir / "captions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing captions file: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                answer = r.get("text", "")
                answer_class = classify_answer(answer)

                rows.append({
                    "condition": run_name,
                    "question_id": r.get("question_id"),
                    "sample_id": r.get("sample_id"),
                    "image": r.get("image"),
                    "target_object": r.get("target_object"),
                    "label": r.get("label"),
                    "answer": answer,
                    "answer_class": answer_class,
                    "hallucinated": answer_class == "yes",
                    "correct_rejection": answer_class == "no",
                    "unclear": answer_class == "other",
                })

    df = pd.DataFrame(rows)

    if len(df) != 522:
        print(f"WARNING: {run_name} has {len(df)} rows, expected 522")

    return df


def load_region_uncertainty(run_name: str, run_dir: Path) -> pd.DataFrame:
    region_dir = run_dir / "region_uncertainty"
    if not region_dir.exists():
        raise FileNotFoundError(f"Missing region_uncertainty folder: {region_dir}")

    rows = []

    for path in sorted(region_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            r = json.load(f)

        sample_id = r.get("sample_id", path.stem)
        mode = r.get("region_mask_mode", run_name)

        base = {
            "condition": run_name,
            "sample_id": sample_id,
            "region_mask_mode": mode,
            "threshold": r.get("threshold"),
            "mean_uncertainty": r.get("mean_uncertainty"),
            "std_uncertainty": r.get("std_uncertainty"),
            "num_uncertain_patch_tokens": r.get("num_uncertain_patch_tokens"),
            "num_suppressed_patch_tokens": r.get("num_suppressed_patch_tokens"),
            "num_patch_tokens": r.get("num_patch_tokens"),
            "num_total_tokens_in_keep_mask": r.get("num_total_tokens_in_keep_mask"),
        }

        regions = r.get("regions", {})
        for region_name in ["removed", "context", "background"]:
            reg = regions.get(region_name, {})
            row = dict(base)
            row.update({
                "region": region_name,
                "region_total": reg.get("total"),
                "region_uncertain": reg.get("uncertain"),
                "region_suppressed": reg.get("suppressed"),
                "uncertainty_density": reg.get("uncertainty_density"),
                "suppression_density": reg.get("suppression_density"),
            })
            rows.append(row)

    df = pd.DataFrame(rows)

    expected = 522 * 3
    if len(df) != expected:
        print(f"WARNING: {run_name} has {len(df)} region rows, expected {expected}")

    return df


def make_per_sample_table(all_captions: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = ["sample_id", "target_object", "label"]

    base = (
        all_captions[metadata_cols]
        .drop_duplicates(subset=["sample_id"])
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    out = base.copy()

    for condition in RUNS.keys():
        sub = all_captions[all_captions["condition"] == condition].copy()
        sub = sub[["sample_id", "answer", "answer_class", "hallucinated", "correct_rejection", "unclear"]]

        sub = sub.rename(columns={
            "answer": f"{condition}_answer",
            "answer_class": f"{condition}_answer_class",
            "hallucinated": f"{condition}_hallucinated",
            "correct_rejection": f"{condition}_correct_rejection",
            "unclear": f"{condition}_unclear",
        })

        out = out.merge(sub, on="sample_id", how="left")

    return out


def make_summary(all_captions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for condition, sub in all_captions.groupby("condition"):
        n = len(sub)
        hallucinated = int(sub["hallucinated"].sum())
        correct = int(sub["correct_rejection"].sum())
        unclear = int(sub["unclear"].sum())

        rows.append({
            "condition": condition,
            "n": n,
            "hallucinated_yes": hallucinated,
            "correct_rejection_no": correct,
            "unclear_other": unclear,
            "hallucination_rate": hallucinated / n if n else None,
            "correct_rejection_rate": correct / n if n else None,
            "unclear_rate": unclear / n if n else None,
        })

    df = pd.DataFrame(rows)

    order = ["none", "all", "removed", "context", "background"]
    df["condition"] = pd.Categorical(df["condition"], categories=order, ordered=True)
    df = df.sort_values("condition").reset_index(drop=True)

    none_rate = float(df.loc[df["condition"] == "none", "hallucination_rate"].iloc[0])

    df["causal_effect_vs_none"] = none_rate - df["hallucination_rate"]

    return df


def make_by_category(all_captions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (condition, target_object), sub in all_captions.groupby(["condition", "target_object"]):
        n = len(sub)
        hallucinated = int(sub["hallucinated"].sum())
        correct = int(sub["correct_rejection"].sum())
        unclear = int(sub["unclear"].sum())

        rows.append({
            "condition": condition,
            "target_object": target_object,
            "n": n,
            "hallucinated_yes": hallucinated,
            "correct_rejection_no": correct,
            "unclear_other": unclear,
            "hallucination_rate": hallucinated / n if n else None,
            "correct_rejection_rate": correct / n if n else None,
            "unclear_rate": unclear / n if n else None,
        })

    df = pd.DataFrame(rows)

    order = ["none", "all", "removed", "context", "background"]
    df["condition"] = pd.Categorical(df["condition"], categories=order, ordered=True)
    df = df.sort_values(["target_object", "condition"]).reset_index(drop=True)

    return df


def make_region_summary(all_regions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["condition", "region"]

    summary = (
        all_regions
        .groupby(group_cols, observed=False)
        .agg(
            n=("sample_id", "count"),
            mean_region_total=("region_total", "mean"),
            mean_uncertain=("region_uncertain", "mean"),
            mean_suppressed=("region_suppressed", "mean"),
            mean_uncertainty_density=("uncertainty_density", "mean"),
            mean_suppression_density=("suppression_density", "mean"),
            mean_num_uncertain_patch_tokens=("num_uncertain_patch_tokens", "mean"),
            mean_num_suppressed_patch_tokens=("num_suppressed_patch_tokens", "mean"),
        )
        .reset_index()
    )

    condition_order = ["none", "all", "removed", "context", "background"]
    region_order = ["removed", "context", "background"]

    summary["condition"] = pd.Categorical(summary["condition"], categories=condition_order, ordered=True)
    summary["region"] = pd.Categorical(summary["region"], categories=region_order, ordered=True)

    return summary.sort_values(["condition", "region"]).reset_index(drop=True)


def plot_hallucination_rates(summary: pd.DataFrame):
    plot_df = summary.copy()
    x = plot_df["condition"].astype(str)
    y = plot_df["hallucination_rate"] * 100

    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.ylabel("Hallucination rate (%)")
    plt.xlabel("Masking condition")
    plt.title("Hallucination rate by masking condition")
    plt.ylim(0, max(100, y.max() + 5))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hallucination_rate_by_condition.png", dpi=200)
    plt.close()


def plot_causal_effects(summary: pd.DataFrame):
    plot_df = summary[summary["condition"].astype(str) != "none"].copy()
    x = plot_df["condition"].astype(str)
    y = plot_df["causal_effect_vs_none"] * 100

    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.axhline(0, linewidth=1)
    plt.ylabel("Causal effect vs none (percentage points)")
    plt.xlabel("Masking condition")
    plt.title("Effect of masking on hallucination")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "causal_effect_by_condition.png", dpi=200)
    plt.close()


def plot_region_suppression(region_summary: pd.DataFrame):
    pivot = region_summary.pivot(index="condition", columns="region", values="mean_suppression_density")

    plt.figure(figsize=(8, 5))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Mean suppression density")
    plt.xlabel("Masking condition")
    plt.title("Mean suppression density by region")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mean_suppression_density_by_region.png", dpi=200)
    plt.close()


def main():
    print("Run 016 — analyzing removed-image evaluation outputs")

    caption_dfs = []
    region_dfs = []

    for condition, run_dir in RUNS.items():
        print(f"Loading {condition}: {run_dir}")
        caption_dfs.append(load_captions(condition, run_dir))
        region_dfs.append(load_region_uncertainty(condition, run_dir))

    all_captions = pd.concat(caption_dfs, ignore_index=True)
    all_regions = pd.concat(region_dfs, ignore_index=True)

    per_sample = make_per_sample_table(all_captions)
    summary = make_summary(all_captions)
    by_category = make_by_category(all_captions)
    region_summary = make_region_summary(all_regions)

    all_captions.to_csv(METRICS_DIR / "removed_eval_all_captions_long.csv", index=False)
    all_regions.to_csv(METRICS_DIR / "removed_eval_region_uncertainty_long.csv", index=False)
    per_sample.to_csv(METRICS_DIR / "removed_eval_per_sample.csv", index=False)
    summary.to_csv(METRICS_DIR / "removed_eval_summary.csv", index=False)
    by_category.to_csv(METRICS_DIR / "removed_eval_by_category.csv", index=False)
    region_summary.to_csv(METRICS_DIR / "removed_eval_region_summary.csv", index=False)

    plot_hallucination_rates(summary)
    plot_causal_effects(summary)
    plot_region_suppression(region_summary)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    print("\nSaved metrics:")
    for path in sorted(METRICS_DIR.glob("*.csv")):
        print(path.relative_to(ROOT))

    print("\nSaved plots:")
    for path in sorted(PLOTS_DIR.glob("*.png")):
        print(path.relative_to(ROOT))

    print("\nRun 016 completed.")


if __name__ == "__main__":
    main()