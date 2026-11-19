# self-fulfilling-forecasts

Code and data for **"Self-Fulfilling Forecasts in a Minimal LLM Market: An Experimental Test of Performative Feedback Strength"** (Beril Demiroglu).

Five persona-prompted LLM agents forecast the price of a market whose realized price is
`P_t = (1 - eps) * F_t + eps * mean(forecasts)`, where `F_t` is an unobserved geometric random walk and
`eps` is the feedback strength. The experiment covers `eps` in {0, 0.2, 0.4, 0.6, 0.8}, 8 seeds and 40 rounds.

## Layout

| Folder | Contents |
|---|---|
| `src/` | experiment code (API calls, personas, round loop) |
| `data/main_experiment_luna.csv` | raw experiment output (one row per eps, seed, round) |
| `data/*.csv` | intermediate tables produced by the analysis scripts |
| `analysis/` | analysis scripts (run from the repo root, in the order below) |
| `results/` | figures used in the paper |

## Reproducing the analysis

```
python3 analysis/h1a_h1b_decomposition.py   # mechanical baseline and residual D^amp (seed level)
python3 analysis/h1b_regression.py          # H1b: direction of amplification
python3 analysis/h2_volatility.py           # H2a / H2b: volatility and magnitude
python3 analysis/h2_robust.py               # HC3 and Spearman checks
python3 analysis/seed_robustness.py         # seed-clustered SEs, seed fixed effects, exact wild cluster bootstrap
python3 analysis/persona_breakdown.py       # persona-level regressions (HC3)
python3 analysis/persona_interaction.py     # corrective vs extrapolative interaction, BH correction (seed-clustered p)
python3 analysis/null_simulation.py         # fixed-rule benchmark (no LLM)
python3 analysis/rule_stability.py          # estimated forecasting rules and their change with eps
python3 analysis/visualize_results.py       # Figure 1
python3 analysis/round_level_plots.py       # Figure 2
```

Requires `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib`.

## Notes

- Volatility and amplification-magnitude regressions use the four non-zero `eps` levels (N = 32).
- Benjamini-Hochberg p-values in `persona_interaction.py` use seed-clustered standard errors; the per-persona
  p-values in `persona_breakdown.py` use HC3.
- The fixed-rule benchmark uses the eight real `F_t` paths from the experiment and 2,000 simulated paths.
  Rule parameters were chosen in advance and are not calibrated to the LLM output.
- The experiment was run with the provider's default (non-zero) temperature. Re-running `src/` will not reproduce the
  numbers exactly; `data/main_experiment_luna.csv` holds the data behind the paper.
