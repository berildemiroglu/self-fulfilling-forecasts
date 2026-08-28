import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/main_experiment_luna.csv")

# Use the same seed (seed=1) across all epsilon values for a fair visual comparison
seed_to_plot = 1

fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
epsilons = [0.0, 0.2, 0.4, 0.6, 0.8]

for ax, eps in zip(axes, epsilons):
    subset = df[(df["epsilon"] == eps) & (df["seed"] == seed_to_plot)].sort_values("round")
    ax.plot(subset["round"], subset["F_t"], label="F_t (fundamental)",
            color="gray", linestyle="--", alpha=0.8)
    ax.plot(subset["round"], subset["P_t"], label="P_t (realized price)",
            color="crimson", linewidth=2)
    ax.set_ylabel("Price")
    ax.set_title(f"epsilon = {eps}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Round")
plt.suptitle(f"F_t vs P_t Across Rounds, by Epsilon (seed={seed_to_plot})", y=1.00, fontsize=14)
plt.tight_layout()
plt.savefig("results/round_level_paths.png", dpi=150, bbox_inches="tight")
print("Saved figure to results/round_level_paths.png")