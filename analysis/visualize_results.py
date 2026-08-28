import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/main_experiment_luna.csv")
volatility = pd.read_csv("data/h2_volatility.csv")
seed_level = pd.read_csv("data/h1a_h1b_seed_level.csv")
persona_data = pd.read_csv("data/persona_breakdown.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Plot 1: Volatility vs epsilon (the strongest result) ---
ax1 = axes[0, 0]
vol_summary = volatility.groupby("epsilon")["volatility"].agg(["mean", "std"])
ax1.errorbar(vol_summary.index, vol_summary["mean"], yerr=vol_summary["std"],
             marker="o", capsize=5, color="darkred", linewidth=2)
ax1.set_xlabel("Epsilon (feedback strength)")
ax1.set_ylabel("Volatility (std of deviation across rounds)")
ax1.set_title("H2a: Volatility Increases Sharply with Epsilon\n(R²=0.833, p<0.001)")
ax1.grid(alpha=0.3)

# --- Plot 2: Residual (H1b, directional) vs epsilon -- showing the null result ---
ax2 = axes[0, 1]
res_summary = seed_level[seed_level["epsilon"] > 0].groupby("epsilon")["residual"].agg(["mean", "std"])
ax2.errorbar(res_summary.index, res_summary["mean"], yerr=res_summary["std"],
             marker="s", capsize=5, color="steelblue", linewidth=2)
ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
ax2.set_xlabel("Epsilon (feedback strength)")
ax2.set_ylabel("Mean residual (directional amplification)")
ax2.set_title("H1b: No Directional Amplification\n(hovers around zero, p=0.903)")
ax2.grid(alpha=0.3)

# --- Plot 3: |residual| (H2b, magnitude) vs epsilon ---
ax3 = axes[1, 0]
seed_level["abs_residual"] = seed_level["residual"].abs()
abs_summary = seed_level[seed_level["epsilon"] > 0].groupby("epsilon")["abs_residual"].agg(["mean", "std"])
ax3.errorbar(abs_summary.index, abs_summary["mean"], yerr=abs_summary["std"],
             marker="^", capsize=5, color="darkorange", linewidth=2)
ax3.set_xlabel("Epsilon (feedback strength)")
ax3.set_ylabel("|Residual| (magnitude of amplification)")
ax3.set_title("H2b: Amplification Magnitude Grows with Epsilon\n(R²=0.338, robust p=0.003)")
ax3.grid(alpha=0.3)

# --- Plot 4: Persona-level |bias| vs epsilon (the disconfirmed hypothesis) ---
ax4 = axes[1, 1]
EXTRAPOLATIVE = ["trend_following", "momentum_sensitive"]
CORRECTIVE = ["mean_reverting", "fundamentals_anchored"]
colors = {"trend_following": "salmon", "momentum_sensitive": "lightcoral",
          "mean_reverting": "steelblue", "fundamentals_anchored": "navy",
          "neutral": "gray"}
for persona in persona_data["persona"].unique():
    subset = persona_data[persona_data["persona"] == persona]
    style = "--" if persona in EXTRAPOLATIVE else ("-" if persona in CORRECTIVE else ":")
    ax4.plot(subset["epsilon"], subset["mean_abs_bias"], marker="o",
             label=persona, color=colors.get(persona, "black"), linestyle=style)
ax4.set_xlabel("Epsilon (feedback strength)")
ax4.set_ylabel("Mean |bias| (forecast - F_t)")
ax4.set_title("Persona Breakdown: Corrective (solid) > Extrapolative (dashed)\n(pre-registered hypothesis disconfirmed)")
ax4.legend(fontsize=8, loc="upper left")
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/main_findings.png", dpi=150, bbox_inches="tight")
print("Saved figure to results/main_findings.png")
plt.show()