import pandas as pd
import numpy as np
import statsmodels.api as sm

volatility = pd.read_csv("data/h2_volatility.csv")
seed_level = pd.read_csv("data/h1a_h1b_seed_level.csv")

# --- H2a robust: volatility ~ epsilon, with HC3 standard errors ---
vol_data = volatility[volatility["epsilon"] > 0]
X = sm.add_constant(vol_data["epsilon"])
y = vol_data["volatility"]
model_vol_robust = sm.OLS(y, X).fit(cov_type="HC3")

print("=== H2a Regression (ROBUST HC3): volatility ~ epsilon ===")
print(model_vol_robust.summary())
print()

# --- H2b robust: abs(residual) ~ epsilon, with HC3 standard errors ---
seed_data = seed_level[seed_level["epsilon"] > 0].copy()
seed_data["abs_residual"] = seed_data["residual"].abs()

X2 = sm.add_constant(seed_data["epsilon"])
y2 = seed_data["abs_residual"]
model_abs_robust = sm.OLS(y2, X2).fit(cov_type="HC3")

print("=== H2b Regression (ROBUST HC3): abs(residual) ~ epsilon ===")
print(model_abs_robust.summary())
print()

# --- Also try a log-log or rank-based check as an additional robustness angle ---
# Spearman correlation (rank-based, distribution-free) as a sanity check
from scipy.stats import spearmanr

rho_vol, p_vol = spearmanr(vol_data["epsilon"], vol_data["volatility"])
rho_abs, p_abs = spearmanr(seed_data["epsilon"], seed_data["abs_residual"])

print("=== Spearman rank correlation (distribution-free robustness check) ===")
print(f"Volatility vs epsilon: rho={rho_vol:.3f}, p={p_vol:.6f}")
print(f"Abs residual vs epsilon: rho={rho_abs:.3f}, p={p_abs:.6f}")