import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
OUT = METRICS / "removed_eval_bootstrap_effects.csv"

df = pd.read_csv(METRICS / "removed_eval_per_sample.csv")

conditions = ["all", "removed", "context", "background"]
n_boot = 10000
rng = np.random.default_rng(42)

rows = []

none = df["none_hallucinated"].astype(int).to_numpy()
n = len(df)

for condition in conditions:
    cond = df[f"{condition}_hallucinated"].astype(int).to_numpy()

    # Effect = HR_none - HR_condition
    observed = none.mean() - cond.mean()

    boot_effects = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        effect = none[idx].mean() - cond[idx].mean()
        boot_effects.append(effect)

    boot_effects = np.array(boot_effects)

    ci_low, ci_high = np.percentile(boot_effects, [2.5, 97.5])

    # approximate two-sided p-value for effect different from zero
    if observed >= 0:
        p = 2 * min((boot_effects <= 0).mean(), (boot_effects >= 0).mean())
    else:
        p = 2 * min((boot_effects >= 0).mean(), (boot_effects <= 0).mean())

    rows.append({
        "condition": condition,
        "observed_effect": observed,
        "observed_effect_percentage_points": observed * 100,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_low_percentage_points": ci_low * 100,
        "ci_high_percentage_points": ci_high * 100,
        "p_value_approx": p,
        "n_samples": n,
        "n_bootstrap": n_boot,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

print("\n=== Bootstrap causal effects vs none ===")
print(out.to_string(index=False))

print(f"\nSaved: {OUT}")