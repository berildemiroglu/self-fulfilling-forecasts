import pandas as pd
import numpy as np
import statsmodels.api as sm

seed_level = pd.read_csv("data/h1a_h1b_seed_level.csv")

# Exclude epsilon=0 (deviation is trivially 0 by construction, no information)
data = seed_level[seed_level["epsilon"] > 0].copy()

print(f"N observations (seed x epsilon, excluding eps=0): {len(data)}")
print()

# --- H1b test: regress observed deviation on epsilon AND the mechanical baseline jointly ---
# A significant residual coefficient on epsilon (after controlling for the
# mechanical baseline) is the evidence for amplification, per the proposal's
# own Layer 4 definition.

X = data[["epsilon", "mechanical_baseline"]]
X = sm.add_constant(X)
y = data["observed_deviation"]

model = sm.OLS(y, X).fit()
print("=== H1b Regression: observed_deviation ~ epsilon + mechanical_baseline ===")
print(model.summary())
print()

# --- Simpler, more direct test: regress the residual itself on epsilon ---
# If residual grows systematically with epsilon, that IS the amplification signal.
X2 = sm.add_constant(data["epsilon"])
y2 = data["residual"]
model2 = sm.OLS(y2, X2).fit()
print("=== Direct test: residual ~ epsilon ===")
print(model2.summary())