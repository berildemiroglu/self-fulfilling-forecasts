import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("data/main_experiment_luna.csv")
seed_level = pd.read_csv("data/h1a_h1b_seed_level.csv")

# --- H2a: Rolling-window volatility per seed per epsilon (proposal's own definition) ---
# Use std of deviation across the 40 rounds, per (epsilon, seed) -- proxy for
# "rolling volatility" aggregated to one value per seed per epsilon.
volatility = df.groupby(["epsilon", "seed"])["deviation"].std().reset_index()
volatility.columns = ["epsilon", "seed", "volatility"]

print("=== Volatility (std of deviation across 40 rounds) by epsilon ===")
print(volatility.groupby("epsilon")["volatility"].agg(["mean", "std", "min", "max"]))
print()

# Test: does volatility increase with epsilon? (excluding eps=0, trivially 0)
vol_data = volatility[volatility["epsilon"] > 0]
X = sm.add_constant(vol_data["epsilon"])
y = vol_data["volatility"]
model_vol = sm.OLS(y, X).fit()
print("=== H2a Regression: volatility ~ epsilon ===")
print(model_vol.summary())
print()

# --- H2b: Absolute residual test (as proposed) ---
# Does the MAGNITUDE of amplification (regardless of direction) grow with epsilon?
seed_data = seed_level[seed_level["epsilon"] > 0].copy()
seed_data["abs_residual"] = seed_data["residual"].abs()

print("=== Absolute residual (|amplification|) by epsilon ===")
print(seed_data.groupby("epsilon")["abs_residual"].agg(["mean", "std"]))
print()

X2 = sm.add_constant(seed_data["epsilon"])
y2 = seed_data["abs_residual"]
model_abs = sm.OLS(y2, X2).fit()
print("=== H2b Regression: abs(residual) ~ epsilon ===")
print(model_abs.summary())

volatility.to_csv("data/h2_volatility.csv", index=False)