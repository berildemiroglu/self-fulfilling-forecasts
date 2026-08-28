import pandas as pd
import numpy as np

df = pd.read_csv("data/main_experiment_luna.csv")

# Step 1: compute each agent's forecast bias at epsilon=0
# (at eps=0, agents' forecasts don't affect P_t, so this bias is "pure" --
#  not yet contaminated by epsilon's mechanical feedback)
df["forecasts_list"] = df["forecasts"].apply(eval)
df["agg_forecast"] = df["forecasts_list"].apply(np.mean)
df["bias"] = df["agg_forecast"] - df["F_t"]

eps0 = df[df["epsilon"] == 0.0]

# Seed-level mean bias at epsilon=0 (this is our mechanical-baseline predictor)
seed_bias_eps0 = eps0.groupby("seed")["bias"].mean()
print("=== Seed-level mean bias at epsilon=0 ===")
print(seed_bias_eps0)
print()

# Step 2: for each (epsilon, seed), compute:
#   - observed mean deviation (actual)
#   - mechanical baseline prediction = epsilon * seed_bias_eps0[seed]
seed_level = df.groupby(["epsilon", "seed"])["deviation"].mean().reset_index()
seed_level.columns = ["epsilon", "seed", "observed_deviation"]

seed_level["mechanical_baseline"] = seed_level.apply(
    lambda row: row["epsilon"] * seed_bias_eps0[row["seed"]], axis=1
)
seed_level["residual"] = seed_level["observed_deviation"] - seed_level["mechanical_baseline"]

print("=== Seed-level decomposition (n=40 observations) ===")
print(seed_level.to_string(index=False))
print()

# Step 3: H1b test -- is the residual (amplification component) significantly
# different from zero, and does it grow with epsilon?
print("=== Residual (amplification) summary by epsilon ===")
print(seed_level.groupby("epsilon")["residual"].agg(["mean", "std"]))

seed_level.to_csv("data/h1a_h1b_seed_level.csv", index=False)
print("\nSaved seed-level decomposition to data/h1a_h1b_seed_level.csv")