import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr

df = pd.read_csv("data/main_experiment_luna.csv")
df["forecasts_list"] = df["forecasts"].apply(eval)

# Persona order matches PERSONAS dict insertion order in the experiment script:
# 0: trend_following, 1: mean_reverting, 2: fundamentals_anchored,
# 3: momentum_sensitive, 4: neutral
PERSONA_NAMES = ["trend_following", "mean_reverting", "fundamentals_anchored",
                  "momentum_sensitive", "neutral"]

for i, name in enumerate(PERSONA_NAMES):
    df[f"forecast_{name}"] = df["forecasts_list"].apply(lambda x: x[i])
    df[f"bias_{name}"] = df[f"forecast_{name}"] - df["F_t"]

# Pre-registered grouping (per Model Spesifikasyonu §6 revision note)
EXTRAPOLATIVE = ["trend_following", "momentum_sensitive"]
CORRECTIVE = ["mean_reverting", "fundamentals_anchored"]
# neutral tracked separately

print("=== Mean |bias| by persona and epsilon ===")
results = []
for name in PERSONA_NAMES:
    df[f"abs_bias_{name}"] = df[f"bias_{name}"].abs()
    grouped = df.groupby("epsilon")[f"abs_bias_{name}"].mean()
    for eps, val in grouped.items():
        results.append({"persona": name, "epsilon": eps, "mean_abs_bias": val})

results_df = pd.DataFrame(results)
pivot = results_df.pivot(index="epsilon", columns="persona", values="mean_abs_bias")
print(pivot.round(3))
print()

# Test the pre-registered hypothesis: does |bias| grow faster with epsilon
# for extrapolative personas than corrective ones?
print("=== Regression: |bias| ~ epsilon, by persona ===")
for name in PERSONA_NAMES:
    X = sm.add_constant(df["epsilon"])
    y = df[f"abs_bias_{name}"]
    model = sm.OLS(y, X).fit(cov_type="HC3")
    coef = model.params["epsilon"]
    pval = model.pvalues["epsilon"]
    r2 = model.rsquared
    group = "EXTRAPOLATIVE" if name in EXTRAPOLATIVE else ("CORRECTIVE" if name in CORRECTIVE else "NEUTRAL")
    print(f"{name:22s} [{group:13s}] coef={coef:.3f}  p={pval:.4f}  R2={r2:.3f}")

print()
print("=== Pre-registered hypothesis check: extrapolative vs corrective growth rate ===")
extrap_coefs = []
correct_coefs = []
for name in PERSONA_NAMES:
    X = sm.add_constant(df["epsilon"])
    y = df[f"abs_bias_{name}"]
    model = sm.OLS(y, X).fit()
    if name in EXTRAPOLATIVE:
        extrap_coefs.append(model.params["epsilon"])
    elif name in CORRECTIVE:
        correct_coefs.append(model.params["epsilon"])

print(f"Mean epsilon-coefficient, extrapolative personas: {np.mean(extrap_coefs):.3f}")
print(f"Mean epsilon-coefficient, corrective personas: {np.mean(correct_coefs):.3f}")
print(f"Ratio (extrapolative/corrective): {np.mean(extrap_coefs)/np.mean(correct_coefs):.2f}x")

results_df.to_csv("data/persona_breakdown.csv", index=False)