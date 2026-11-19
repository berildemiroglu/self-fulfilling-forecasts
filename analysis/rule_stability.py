"""
Do the LLM forecasting rules change with epsilon?
For each persona and epsilon, regress (forecast - P_{t-1}) on
  dP  = P_{t-1} - P_{t-2}          (trend response)
  gap5 = mean(P_{t-5..t-1}) - P_{t-1}   (short-run reversion response)
  gapL = mean(P_{1..t-1}) - P_{t-1}     (long-run anchor response)
pooled over seeds and rounds 4..40. If the coefficients are stable across
epsilon, the persona's rule did not change; amplification is then lag, not behavior.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
NAMES = ["trend_following", "mean_reverting", "fundamentals_anchored", "momentum_sensitive", "neutral"]
df = pd.read_csv("data/main_experiment_luna.csv"); df["f"] = df.forecasts.apply(eval)
rows = []
for (e, s), g in df.sort_values("round").groupby(["epsilon", "seed"]):
    P = g.P_t.values; F = np.vstack(g.f.values)
    for t in range(3, 40):                    # 0-based t: needs P[t-1], P[t-2], and >=3 history
        h = P[:t]
        for i, n in enumerate(NAMES):
            rows.append(dict(eps=e, seed=s, persona=n, y=F[t, i] - h[-1], dP=h[-1] - h[-2],
                             gap5=h[-5:].mean() - h[-1], gapL=h.mean() - h[-1]))
d = pd.DataFrame(rows)
out = []
for (n, e), g in d.groupby(["persona", "eps"]):
    m = sm.OLS(g.y, g[["dP", "gap5", "gapL"]]).fit()
    out.append(dict(persona=n, eps=e, dP=m.params.dP, gap5=m.params.gap5, gapL=m.params.gapL, R2=m.rsquared))
o = pd.DataFrame(out)
for c in ["dP", "gap5", "gapL", "R2"]:
    print(f"\n{c}"); print(o.pivot(index="persona", columns="eps", values=c).round(2))

# ---------------------------------------------------------------------------
# Part 2: does each response coefficient change with epsilon? (formal test)
# y = (b0 + b1*eps)*dP + (c0 + c1*eps)*gap5 + (g0 + g1*eps)*gapL + error
# Seed-clustered SEs (only 8 clusters: treat p-values as approximate).
# The coefficient of interest is the interaction slope (b1, c1, g1).
# ---------------------------------------------------------------------------
print("\n\n=== Interaction slopes: change in response coefficient per unit of epsilon ===")
rows = []
for n, g in d.groupby("persona"):
    g = g.copy()
    for v in ["dP", "gap5", "gapL"]:
        g[v + "_x_eps"] = g[v] * g.eps
    X = g[["dP", "gap5", "gapL", "dP_x_eps", "gap5_x_eps", "gapL_x_eps"]]
    m = sm.OLS(g.y, X).fit(cov_type="cluster", cov_kwds={"groups": g.seed})
    for v in ["dP", "gap5", "gapL"]:
        k = v + "_x_eps"
        rows.append(dict(persona=n, response=v, level_at_eps0=m.params[v], slope_per_eps=m.params[k],
                         se=m.bse[k], p=m.pvalues[k]))
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
res.to_csv("data/rule_stability_interaction.csv", index=False)
