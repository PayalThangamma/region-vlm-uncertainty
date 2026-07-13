
import csv
from pathlib import Path

ROOT = Path("outputs") / "metrics"

MODELS = {
    "llava7b": ROOT / "llava7b" / "removed_eval_all_captions_long.csv",
    "llava13b": ROOT / "llava13b" / "removed_eval_all_captions_long.csv",
}

MODES = ["all", "removed", "context", "background"]


def classify_answer(text):
    t = (text or "").strip().lower()

    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"

    first = t[:80]
    if "yes" in first:
        return "yes"
    if "no" in first:
        return "no"

    return "unknown"


def pick_column(fieldnames, candidates):
    lower_map = {c.lower(): c for c in fieldnames}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def load_long_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    print(f"\nLoaded: {path}")
    print("Columns:", fieldnames)

    condition_col = pick_column(
        fieldnames,
        ["condition", "mode", "region_mask_mode", "mask_mode"],
    )

    qid_col = pick_column(
        fieldnames,
        ["question_id", "qid", "id"],
    )

    sample_col = pick_column(
        fieldnames,
        ["sample_id", "sample", "image_id"],
    )

    text_col = pick_column(
        fieldnames,
        ["text", "answer_text", "response", "caption", "model_output", "output"],
    )

    answer_col = pick_column(
        fieldnames,
        ["answer", "pred_answer", "prediction", "pred", "yes_no"],
    )

    if condition_col is None:
        raise RuntimeError(f"Could not find condition column in {path}")

    if qid_col is None and sample_col is None:
        raise RuntimeError(f"Could not find question_id or sample_id column in {path}")

    if text_col is None and answer_col is None:
        raise RuntimeError(f"Could not find text/answer column in {path}")

    data = {}

    for r in rows:
        condition = (r.get(condition_col) or "").strip()

        if condition == "":
            continue

        qid = (r.get(qid_col) or "").strip() if qid_col else ""
        sample_id = (r.get(sample_col) or "").strip() if sample_col else ""

        # Prefer question_id. Fallback to sample_id.
        key = qid if qid else sample_id

        if key == "":
            continue

        if answer_col and (r.get(answer_col) or "").strip():
            ans = classify_answer(r.get(answer_col, ""))
        else:
            ans = classify_answer(r.get(text_col, ""))

        data[(condition, key)] = ans

    return data


def analyze_model(model_name, csv_path):
    data = load_long_csv(csv_path)

    none_keys = sorted(k for (condition, k) in data if condition == "none")

    if not none_keys:
        raise RuntimeError(f"No condition='none' rows found for {model_name}")

    out_path = ROOT / model_name / "removed_eval_answer_flips.csv"

    lines = [
        "condition,total,yes_to_no,no_to_yes,unchanged_yes,unchanged_no,unknown_changes,net_hallucination_reduction"
    ]

    print()
    print(f"===== {model_name} answer flips =====")
    print(lines[0])

    for mode in MODES:
        yes_to_no = 0
        no_to_yes = 0
        unchanged_yes = 0
        unchanged_no = 0
        unknown_changes = 0
        missing = 0

        for key in none_keys:
            a = data.get(("none", key))
            b = data.get((mode, key))

            if b is None:
                missing += 1
                continue

            if a == "yes" and b == "no":
                yes_to_no += 1
            elif a == "no" and b == "yes":
                no_to_yes += 1
            elif a == "yes" and b == "yes":
                unchanged_yes += 1
            elif a == "no" and b == "no":
                unchanged_no += 1
            else:
                unknown_changes += 1

        total = len(none_keys)
        net = yes_to_no - no_to_yes

        row = (
            f"{mode},{total},{yes_to_no},{no_to_yes},"
            f"{unchanged_yes},{unchanged_no},{unknown_changes},{net}"
        )

        print(row)

        if missing:
            print(f"WARNING: {model_name} {mode} missing rows: {missing}")

        lines.append(row)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")


def main():
    for model_name, csv_path in MODELS.items():
        analyze_model(model_name, csv_path)


if __name__ == "__main__":
    main()