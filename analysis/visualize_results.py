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
ax1.set_xlabel(r"$\varepsilon$ (feedback strength)")
ax1.set_ylabel(r"Volatility (SD of $D_t$ across rounds)")
ax1.set_title(r"H2a: volatility increases with $\varepsilon$" "\n" r"($R^2$=0.833, $p$<0.001; $\varepsilon$>0, N=32)")
ax1.grid(alpha=0.3)

# --- Plot 2: Residual (H1b, directional) vs epsilon -- showing the null result ---
ax2 = axes[0, 1]
res_summary = seed_level[seed_level["epsilon"] > 0].groupby("epsilon")["residual"].agg(["mean", "std"])
ax2.errorbar(res_summary.index, res_summary["mean"], yerr=res_summary["std"],
             marker="s", capsize=5, color="steelblue", linewidth=2)
ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
ax2.set_xlabel(r"$\varepsilon$ (feedback strength)")
ax2.set_ylabel(r"Mean residual $D_t^{amp}$ (direction)")
ax2.set_title("H1b: no directional amplification\n" r"(mean residual near zero, $p$=0.903)")
ax2.grid(alpha=0.3)

# --- Plot 3: |residual| (H2b, magnitude) vs epsilon ---
ax3 = axes[1, 0]
seed_level["abs_residual"] = seed_level["residual"].abs()
abs_summary = seed_level[seed_level["epsilon"] > 0].groupby("epsilon")["abs_residual"].agg(["mean", "std"])
ax3.errorbar(abs_summary.index, abs_summary["mean"], yerr=abs_summary["std"],
             marker="^", capsize=5, color="darkorange", linewidth=2)
ax3.set_xlabel(r"$\varepsilon$ (feedback strength)")
ax3.set_ylabel(r"$|\overline{D^{amp}}|$ (seed-level mean, magnitude)")
ax3.set_title(r"H2b: magnitude of amplification grows with $\varepsilon$" "\n" r"($R^2$=0.338, robust $p$=0.003)")
ax3.grid(alpha=0.3)

# --- Plot 4: Persona-level |bias| vs epsilon (the disconfirmed hypothesis) ---
ax4 = axes[1, 1]
EXTRAPOLATIVE = ["trend_following", "momentum_sensitive"]
CORRECTIVE = ["mean_reverting", "fundamentals_anchored"]
colors = {"trend_following": "salmon", "momentum_sensitive": "firebrick",
          "mean_reverting": "steelblue", "fundamentals_anchored": "navy",
          "neutral": "darkgreen"}
for persona in persona_data["persona"].unique():
    subset = persona_data[persona_data["persona"] == persona]
    style = "--" if persona in EXTRAPOLATIVE else ("-" if persona in CORRECTIVE else ":")
    ax4.plot(subset["epsilon"], subset["mean_abs_bias"], marker=("D" if persona=="neutral" else "o"), linewidth=(2.5 if persona=="neutral" else 1.5),
             label=persona.replace("_", " "), color=colors.get(persona, "black"), linestyle=style)
ax4.set_xlabel(r"$\varepsilon$ (feedback strength)")
ax4.set_ylabel(r"Mean $|f - F_t|$")
ax4.set_title("Persona breakdown: corrective (solid), extrapolative (dashed),\nneutral (dotted); hypothesis not supported")
ax4.legend(fontsize=8, loc="upper left")
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/main_findings.png", dpi=150, bbox_inches="tight")
print("Saved figure to results/main_findings.png")
