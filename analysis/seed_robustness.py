"""
Addresses the reviewer's core methodological concern:

    "8 seeds, each reused across multiple epsilon levels -> the 32
    seed/epsilon observations are NOT independent. HC3 corrects for
    heteroskedasticity but not within-seed serial correlation."

This script re-runs the H1b and H2 regressions three ways, from least to
most conservative, so the paper can report whichever level of rigor the
venue expects (or all three, as a robustness table):

    1. Cluster-robust standard errors, clustered by seed
       (accounts for within-seed correlation, but asymptotic -- with only
       8 clusters the usual normal/t approximation can be unreliable)
    2. Seed fixed effects (absorbs any seed-level baseline that doesn't
       interact with epsilon)
    3. EXACT wild cluster bootstrap (Cameron, Gelbach & Miller 2008)
       -- with exactly 8 clusters there are only 2^8 = 256 possible sign
       patterns, so we enumerate ALL of them and get an *exact* small-sample
       p-value rather than an asymptotic approximation. This is the
       reviewer's "permutation / sign-flip" suggestion, done exhaustively.

Run this from the repo root (expects data/h1a_h1b_seed_level.csv and
data/h2_volatility.csv to already exist -- i.e. run
analysis/h1a_h1b_decomposition.py and analysis/h2_volatility.py first).
"""
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option("display.width", 100)


# ---------------------------------------------------------------------------
# Exact wild cluster bootstrap (Rademacher weights, full enumeration)
# ---------------------------------------------------------------------------
def exact_wild_cluster_bootstrap(data, y_col, x_col, cluster_col, fixed_effects=False):
    """
    Test H0: beta_x = 0 in  y ~ x (+ cluster fixed effects)
    using the exact wild cluster bootstrap: since there are few clusters,
    every one of the 2^G possible Rademacher sign patterns is enumerated
    exactly (no Monte Carlo approximation).

    Returns a dict with the original t-stat, the exact bootstrap p-value,
    and the number of sign patterns evaluated.
    """
    clusters = sorted(data[cluster_col].unique())
    G = len(clusters)

    # --- Step 1: fit the model actually of interest, get the real t-stat ---
    if fixed_effects:
        formula = f"{y_col} ~ {x_col} + C({cluster_col})"
    else:
        formula = f"{y_col} ~ {x_col}"
    full_model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col]}
    )
    t_obs = full_model.tvalues[x_col]

    # --- Step 2: fit the RESTRICTED model (imposing beta_x = 0) to get
    #     fitted values + residuals to resample from ---
    if fixed_effects:
        restricted_formula = f"{y_col} ~ C({cluster_col})"
    else:
        restricted_formula = f"{y_col} ~ 1"
    restricted = smf.ols(restricted_formula, data=data).fit()
    fitted_restricted = restricted.fittedvalues
    resid_restricted = restricted.resid

    # --- Step 3: enumerate all 2^G Rademacher sign patterns exactly ---
    boot_t = []
    for signs in itertools.product([-1, 1], repeat=G):
        sign_map = dict(zip(clusters, signs))
        weight = data[cluster_col].map(sign_map).values
        y_star = fitted_restricted.values + weight * resid_restricted.values

        boot_data = data.copy()
        boot_data["_y_star"] = y_star
        boot_formula = formula.replace(f"{y_col} ~", "_y_star ~")
        boot_model = smf.ols(boot_formula, data=boot_data).fit(
            cov_type="cluster", cov_kwds={"groups": boot_data[cluster_col]}
        )
        boot_t.append(boot_model.tvalues[x_col])

    boot_t = np.array(boot_t)
    # two-sided exact p-value: fraction of bootstrap |t*| at least as extreme
    p_exact = np.mean(np.abs(boot_t) >= np.abs(t_obs))

    return {
        "t_obs": t_obs,
        "coef_obs": full_model.params[x_col],
        "cluster_robust_p": full_model.pvalues[x_col],
        "n_sign_patterns": len(boot_t),
        "p_exact_wild_cluster": p_exact,
    }


def run_all_tests(df, y_col, label):
    print(f"\n{'=' * 70}")
    print(f"  {label}   (y = {y_col})")
    print(f"{'=' * 70}")

    # 1. Cluster-robust SE (clustered by seed)
    X = sm.add_constant(df["epsilon"])
    m_cluster = sm.OLS(df[y_col], X).fit(
        cov_type="cluster", cov_kwds={"groups": df["seed"]}
    )
    print("\n--- (1) Cluster-robust SE (clustered by seed) ---")
    print(f"epsilon coef = {m_cluster.params['epsilon']:.4f}   "
          f"cluster-robust p = {m_cluster.pvalues['epsilon']:.4f}")

    # 2. Seed fixed effects
    m_fe = smf.ols(f"{y_col} ~ epsilon + C(seed)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["seed"]}
    )
    print("\n--- (2) Seed fixed effects + cluster-robust SE ---")
    print(f"epsilon coef = {m_fe.params['epsilon']:.4f}   "
          f"p = {m_fe.pvalues['epsilon']:.4f}")

    # 3. Exact wild cluster bootstrap (no fixed effects -- matches original spec)
    result = exact_wild_cluster_bootstrap(df, y_col, "epsilon", "seed", fixed_effects=False)
    print("\n--- (3) EXACT wild cluster bootstrap (256 sign patterns, no FE) ---")
    print(f"observed t = {result['t_obs']:.4f}")
    print(f"exact bootstrap p-value = {result['p_exact_wild_cluster']:.4f}")
    print(f"(for comparison, cluster-robust asymptotic p = {result['cluster_robust_p']:.4f})")

    # 3b. Same, but with seed fixed effects included in both restricted/full model
    result_fe = exact_wild_cluster_bootstrap(df, y_col, "epsilon", "seed", fixed_effects=True)
    print("\n--- (3b) EXACT wild cluster bootstrap, WITH seed fixed effects ---")
    print(f"observed t = {result_fe['t_obs']:.4f}")
    print(f"exact bootstrap p-value = {result_fe['p_exact_wild_cluster']:.4f}")

    return {
        "cluster_robust": m_cluster,
        "fixed_effects": m_fe,
        "wild_bootstrap": result,
        "wild_bootstrap_fe": result_fe,
    }


if __name__ == "__main__":
    seed_level = pd.read_csv("data/h1a_h1b_seed_level.csv")
    volatility = pd.read_csv("data/h2_volatility.csv")

    # exclude epsilon=0 (trivially zero by construction, no information)
    seed_data = seed_level[seed_level["epsilon"] > 0].copy()
    seed_data["abs_residual"] = seed_data["residual"].abs()
    vol_data = volatility[volatility["epsilon"] > 0].copy()

    print(f"N seed-level observations (epsilon > 0): {len(seed_data)}  "
          f"({seed_data['seed'].nunique()} seeds x {seed_data['epsilon'].nunique()} epsilons)")

    results = {}
    results["H1b_residual"] = run_all_tests(seed_data, "residual", "H1b: directional amplification")
    results["H2b_abs_residual"] = run_all_tests(seed_data, "abs_residual", "H2b: magnitude of amplification")
    results["H2a_volatility"] = run_all_tests(vol_data, "volatility", "H2a: volatility")

    print(f"\n{'=' * 70}")
    print("SUMMARY -- exact wild cluster bootstrap p-values (no FE)")
    print(f"{'=' * 70}")
    for name, res in results.items():
        p = res["wild_bootstrap"]["p_exact_wild_cluster"]
        print(f"  {name:20s}: p = {p:.4f}")
