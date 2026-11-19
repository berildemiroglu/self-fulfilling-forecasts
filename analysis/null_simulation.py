"""
Null benchmark: fixed, non-LLM forecasting rules in the same market.

Question: how much of the LLM results (volatility, |D_amp|, persona-level
eps-sensitivity) would appear even if agents followed FIXED rules that never
change with epsilon? Any D_amp here is pure recursive price lag, because the
rules themselves do not depend on epsilon.

Run from the repo root:  python3 analysis/null_simulation.py
Reads data/main_experiment_luna.csv (only to reuse the 8 real F_t paths).
"""
import numpy as np, pandas as pd, statsmodels.api as sm

EPS = [0.0, 0.2, 0.4, 0.6, 0.8]
T = 40

# Each rule maps the observed price history P_1..P_{t-1} (a list, may be empty)
# to a forecast. Round 1 (no history): 100, as the LLM prompt requests.
def naive(h):    return h[-1]
def trend(h):    return h[-1] + (h[-1] - h[-2]) if len(h) >= 2 else h[-1]
def momentum(h): return h[-1] + 0.5 * (h[-1] - h[-2]) if len(h) >= 2 else h[-1]
def meanrev(h):  # partial correction toward the average of the last 5 prices
    return h[-1] + 0.5 * (np.mean(h[-5:]) - h[-1])
def anchored(h): # long-run anchor = mean of the entire history, small weight on latest
    return 0.8 * np.mean(h) + 0.2 * h[-1]

RULES = {"trend_following": trend, "mean_reverting": meanrev,
         "fundamentals_anchored": anchored, "momentum_sensitive": momentum,
         "neutral(naive)": naive}

def run(F, eps, rules):
    """One market run. Returns prices P and the (T x n_rules) forecast matrix."""
    P, f_all, hist = [], [], []
    for t in range(T):
        f = [100.0 if not hist else r(hist) for r in rules]
        p = (1 - eps) * F[t] + eps * np.mean(f)      # same-round forecast enters P_t
        hist.append(p); P.append(p); f_all.append(f)
    return np.array(P), np.array(f_all)

def metrics(F, rules):
    """Per-eps seed-level metrics for one F path."""
    P0, f0 = run(F, 0.0, rules)
    B0 = f0.mean(axis=1) - F                         # bias at eps=0
    out = []
    for e in EPS:
        P, f = run(F, e, rules)
        D = P - F; Damp = D - e * B0
        out.append(dict(eps=e, sd_D=D.std(ddof=1), sd_Damp=Damp.std(ddof=1),
                        excess_sd=D.std(ddof=1) - e * B0.std(ddof=1),
                        mean_Damp=Damp.mean(), abs_mean_Damp=abs(Damp.mean()),
                        persona_abs_bias=np.abs(f - F[:, None]).mean(axis=0)))
    return out

def slope(df, y):
    d = df[df.eps > 0]
    m = sm.OLS(d[y], sm.add_constant(d.eps)).fit(cov_type="HC3")
    return m.params["eps"], m.rsquared

def table(F_paths, rules, label):
    rows = []
    for s, F in enumerate(F_paths):
        for r in metrics(F, rules):
            r["seed"] = s; rows.append(r)
    df = pd.DataFrame(rows)
    res = {y: slope(df, y) for y in ["sd_D", "sd_Damp", "excess_sd", "abs_mean_Damp", "mean_Damp"]}
    print(f"\n{label}")
    for y, (b, r2) in res.items():
        print(f"  {y:14s} beta={b:7.3f}  R2={r2:5.3f}")
    return df

if __name__ == "__main__":
    # (1) the 8 real fundamental paths from the experiment
    raw = pd.read_csv("data/main_experiment_luna.csv")
    real = [g.sort_values("round").F_t.values
            for _, g in raw[raw.epsilon == 0].groupby("seed")]

    print("=== LLM, for reference (from the paper) ===")
    print("  sd_D           beta= 3.144  R2=0.833")
    print("  abs_mean_Damp  beta= 1.862  R2=0.338   (HC3 p=0.003)")
    print("  mean_Damp      beta= 0.084  R2=0.001   (H1b)")

    mixed = list(RULES.values())
    table(real, mixed, "FIXED RULES, mixed 5-rule population, 8 real F_t paths")
    for name, rule in RULES.items():
        table(real, [rule] * 5, f"FIXED RULES, homogeneous '{name}' population, 8 real F_t paths")

    # (2) many simulated F paths -> expected value without 8-seed noise
    rng = np.random.default_rng(0)
    sim = [100 * np.exp(np.cumsum(0.001 + 0.02 * rng.standard_normal(T))) for _ in range(2000)]
    df = table(sim, mixed, "FIXED RULES, mixed population, 2000 simulated F_t paths (expected slopes)")

    # (3) persona-level: mean |f - F_t| by eps, per rule (real paths)
    print("\nPersona-level mean |forecast - F_t| by eps (mixed population, 8 real paths)")
    rows = []
    for F in real:
        for r in metrics(F, mixed):
            for i, n in enumerate(RULES):
                rows.append(dict(eps=r["eps"], persona=n, v=r["persona_abs_bias"][i]))
    pv = pd.DataFrame(rows).groupby(["persona", "eps"]).v.mean().unstack()
    print(pv.round(2))
    print("slope on eps (OLS, all 5 eps levels):")
    for n in RULES:
        print(f"  {n:22s} {np.polyfit(pv.columns.values, pv.loc[n].values, 1)[0]:6.3f}")
