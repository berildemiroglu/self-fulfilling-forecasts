"""
Addresses two reviewer points at once:

1. "Corrective coef (1.311) > extrapolative coef (0.760) does not by
    itself show the two groups are statistically different. Test this
    directly with an epsilon x persona_type interaction term."

2. "Five personas / hypotheses tested -> add a multiple testing
    correction."

Part A builds a single long-format panel (one row per persona per round)
and fits ONE regression with an epsilon x persona_group interaction,
cluster-robust by seed. This directly tests whether the slope of
|bias| ~ epsilon differs across persona groups -- exactly what the
per-persona regressions in the original script could not test.

Part B takes the five original per-persona p-values and applies
Benjamini-Hochberg FDR correction (and reports Bonferroni too, for
comparison).

Run from the repo root; expects data/main_experiment_luna.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

PERSONA_NAMES = ["trend_following", "mean_reverting", "fundamentals_anchored",
                  "momentum_sensitive", "neutral"]
EXTRAPOLATIVE = {"trend_following", "momentum_sensitive"}
CORRECTIVE = {"mean_reverting", "fundamentals_anchored"}


def persona_group(name):
    if name in EXTRAPOLATIVE:
        return "extrapolative"
    if name in CORRECTIVE:
        return "corrective"
    return "neutral"


def build_long_panel(df):
    """One row per (epsilon, seed, round, persona) with abs_bias and group."""
    df = df.copy()
    df["forecasts_list"] = df["forecasts"].apply(eval)

    long_rows = []
    for i, name in enumerate(PERSONA_NAMES):
        sub = df[["epsilon", "seed", "round", "F_t"]].copy()
        sub["persona"] = name
        sub["persona_group"] = persona_group(name)
        sub["forecast"] = df["forecasts_list"].apply(lambda x, i=i: x[i])
        sub["abs_bias"] = (sub["forecast"] - sub["F_t"]).abs()
        long_rows.append(sub)

    return pd.concat(long_rows, ignore_index=True)


def part_a_interaction_test(long_df):
    print("=" * 70)
    print("PART A -- epsilon x persona_group interaction (cluster-robust by seed)")
    print("=" * 70)

    # neutral is the reference group; corrective and extrapolative get
    # their own slope + intercept shift relative to it
    model = smf.ols(
        "abs_bias ~ epsilon * C(persona_group, Treatment(reference='neutral'))",
        data=long_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": long_df["seed"]})

    print(model.summary())
    print()

    # Direct test: is the CORRECTIVE slope different from the EXTRAPOLATIVE slope?
    # Re-fit with extrapolative as reference so we can read the corrective
    # coefficient directly as "corrective slope minus extrapolative slope".
    model_extrap_ref = smf.ols(
        "abs_bias ~ epsilon * C(persona_group, Treatment(reference='extrapolative'))",
        data=long_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": long_df["seed"]})

    interaction_term = "epsilon:C(persona_group, Treatment(reference='extrapolative'))[T.corrective]"
    coef = model_extrap_ref.params.get(interaction_term)
    pval = model_extrap_ref.pvalues.get(interaction_term)

    print("-" * 70)
    print("DIRECT TEST: corrective slope - extrapolative slope")
    print("-" * 70)
    if coef is not None:
        print(f"difference in epsilon-slope (corrective - extrapolative) = {coef:.4f}")
        print(f"p-value (cluster-robust, clustered by seed) = {pval:.4f}")
        if pval < 0.05:
            print("-> statistically significant difference between groups")
        else:
            print("-> NOT statistically significant: cannot reject that the two "
                  "groups respond equally to epsilon, despite the raw coefficient gap")
    else:
        print("Could not locate interaction term -- check patsy term names in "
              "model_extrap_ref.params.index")

    return model, model_extrap_ref


def part_b_multiple_testing(df):
    print()
    print("=" * 70)
    print("PART B -- multiple testing correction across the 5 persona regressions")
    print("=" * 70)

    df = df.copy()
    df["forecasts_list"] = df["forecasts"].apply(eval)

    rows = []
    for i, name in enumerate(PERSONA_NAMES):
        df[f"forecast_{name}"] = df["forecasts_list"].apply(lambda x, i=i: x[i])
        df[f"abs_bias_{name}"] = (df[f"forecast_{name}"] - df["F_t"]).abs()

        X = sm.add_constant(df["epsilon"])
        y = df[f"abs_bias_{name}"]
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["seed"]})
        rows.append({
            "persona": name,
            "group": persona_group(name),
            "coef": model.params["epsilon"],
            "p_raw": model.pvalues["epsilon"],
        })

    results = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR
    reject_bh, p_bh, _, _ = multipletests(results["p_raw"], alpha=0.05, method="fdr_bh")
    results["p_bh"] = p_bh
    results["significant_bh"] = reject_bh

    # Bonferroni, for comparison
    reject_bonf, p_bonf, _, _ = multipletests(results["p_raw"], alpha=0.05, method="bonferroni")
    results["p_bonferroni"] = p_bonf
    results["significant_bonferroni"] = reject_bonf

    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    df = pd.read_csv("data/main_experiment_luna.csv")

    long_panel = build_long_panel(df)
    part_a_interaction_test(long_panel)
    results = part_b_multiple_testing(df)

    results.to_csv("data/persona_interaction_results.csv", index=False)
    print("\nSaved to data/persona_interaction_results.csv")
