# %% [markdown]
# # Week 2 -- Rule Backtesting Fundamentals
#
# **Concept:** Evaluate existing rule performance using historical data.
#
# **Topics**
#
# * Retrospective testing
# * Sample design
# * Selection bias
# * Coverage analysis
# * Alert yield
#
# **Scenario.** Rule: monthly outbound wires > £50k.
#
# **Exercise.** Calculate alert count, true positives and precision at
# thresholds of £25k, £50k, £75k and £100k.
#
# **Success criteria.** Identify the threshold that maximises risk detection
# under a fixed alert volume.

# %%
# --- path bootstrap ---
import pathlib
import sys

for _p in pathlib.Path(__file__ if "__file__" in globals() else "x").resolve().parents:
    if (_p / "src" / "tmtuning").is_dir():
        sys.path.insert(0, str(_p / "src"))
        break

import numpy as np
import pandas as pd

from tmtuning import (answer, banner, generate_population, optimise_threshold, show,
                      spec_population, threshold_sweep)

# %% [markdown]
# ## 1. The synthetic population, exactly as the spec defines it

# %%
banner("1. SYNTHETIC POPULATION (SPEC VERSION)")

np.random.seed(42)
n = 5000

df = pd.DataFrame({
    "amount": np.random.gamma(2, 20000, n),
    "case": np.random.binomial(1, 0.04, n),
})

show(df.describe().T.reset_index().rename(columns={"index": "column"}),
     "Population summary")
print(f"\nBase rate: {df['case'].mean():.2%}   Median amount: £{df['amount'].median():,.0f}")

# %% [markdown]
# ## 2. The exercise: alert count, true positives and precision at each threshold

# %%
banner("2. THRESHOLD SWEEP AT THE FOUR SPECIFIED THRESHOLDS")

results = []
for threshold in [25_000, 50_000, 75_000, 100_000]:
    alerts = df["amount"] > threshold
    alert_count = int(alerts.sum())
    true_positives = int(df.loc[alerts, "case"].sum())
    results.append({
        "threshold": threshold,
        "alert_count": alert_count,
        "true_positives": true_positives,
        "precision": true_positives / alert_count if alert_count else np.nan,
        "recall": true_positives / df["case"].sum(),
    })

show(pd.DataFrame(results), "Alert count, true positives and precision by threshold")

# %% [markdown]
# ## 3. Read the result before you act on it
#
# This is the most important cell of the week.

# %%
banner("3. WHAT THIS RESULT ACTUALLY SHOWS")

sweep = threshold_sweep(df, "amount", "case", n_thresholds=25)

# Restrict to thresholds with enough alerts to estimate precision at all. At the
# extreme tail a handful of alerts makes precision swing between 0% and 20% on
# one record -- that is sampling noise, not signal, and quoting it as the range
# would disguise how flat the curve really is.
MIN_ALERTS = 100
stable = sweep[sweep["alerts"] >= MIN_ALERTS]
print(f"Precision across {len(stable)} thresholds with >= {MIN_ALERTS} alerts, "
      f"spanning £{stable['threshold'].min():,.0f} to £{stable['threshold'].max():,.0f}:")
print(f"  min {stable['precision'].min():.2%}   "
      f"max {stable['precision'].max():.2%}   "
      f"population base rate {df['case'].mean():.2%}")
print(f"\n  (Thresholds above £{stable['threshold'].max():,.0f} are excluded: fewer than "
      f"{MIN_ALERTS}\n   alerts each, so their precision is noise. Small-sample tails are where\n"
      f"   over-tightened thresholds get their misleading evidence.)")

print("""
Precision is flat, and it is flat AT the base rate. Raising the threshold
buys no yield at all while recall collapses.

That is not a tuning failure -- it is the correct answer to the data. In this
generator `case` is drawn from a fixed binomial with no reference to `amount`:

    "case": np.random.binomial(1, 0.04, n)

Risk and transaction value are statistically independent, so no threshold on
`amount` can separate cases from non-cases. The flat line is the finding.
""")

answer("You run this sweep on real data and precision is flat. What do you conclude?",
       """
That the variable you are thresholding carries no risk signal in this
sample. Three possible causes, and they need different responses:

  1. The variable genuinely does not discriminate for this typology.
     -> The rule needs different logic, not a different threshold.
  2. The label set is too sparse or too noisy to show the signal.
     -> Fix the labels before tuning; widen the observation window.
  3. The sample is biased (see section 5) so the signal is masked.
     -> Fix the sample design and re-run.

What you must NOT do is keep sliding the threshold until some number looks
acceptable. A flat precision curve means there is nothing to tune, and a
threshold picked off a flat curve is picked at random however confident the
paper sounds.
""")

# %% [markdown]
# ## 4. Coverage analysis
#
# Alert yield tells you about the alerts you raised. Coverage tells you about
# the risk you did not.

# %%
banner("4. COVERAGE ANALYSIS")

coverage = []
for threshold in [25_000, 50_000, 75_000, 100_000]:
    alerts = df["amount"] > threshold
    cases_caught = int(df.loc[alerts, "case"].sum())
    cases_missed = int(df.loc[~alerts, "case"].sum())
    coverage.append({
        "threshold": threshold,
        "population_covered": alerts.mean(),
        "cases_caught": cases_caught,
        "cases_missed": cases_missed,
        "coverage_of_risk": cases_caught / df["case"].sum(),
    })
show(pd.DataFrame(coverage), "Coverage: what share of known risk does each threshold reach?")

print("\nThe `cases_missed` column is the one a regulator reads first.")
print("It is also the one that is systematically understated in real data,")
print("because cases below the line rarely got investigated and so rarely")
print("acquired a label. Week 6 puts a confidence interval around it.")

# %% [markdown]
# ## 5. Sample design and selection bias

# %%
banner("5. SAMPLE DESIGN AND SELECTION BIAS")

bias = pd.DataFrame([
    ("Alert-only sample", "Backtest using only records that alerted historically",
     "Cannot measure false negatives at all -- recall is undefined, not 100%",
     "Sample the full scored population, not the alert table"),
    ("Investigated-only labels", "Treat 'no SAR' as 'not a case'",
     "Below-the-line risk is invisible, inflating measured recall",
     "Below-the-line sampling with a confidence interval (Week 6)"),
    ("Survivorship", "Exclude customers exited during the period",
     "Removes exactly the highest-risk customers from the sample",
     "Include exited customers; flag exit reason"),
    ("Single-period sample", "One month of data, extrapolated",
     "Seasonality and drift read as rule performance",
     "Span >= 12 months; test periods separately (Week 7)"),
    ("Post-hoc threshold choice", "Pick the threshold, then pick the window that supports it",
     "Overfits to noise; will not hold out of time",
     "Fix the window first; validate out of time (Week 3)"),
], columns=["Bias", "What it looks like", "What it does to your numbers", "Mitigation"])
show(bias, "Five ways a backtest sample lies to you")

# %% [markdown]
# ## 6. Success criteria: maximise detection under a fixed alert volume
#
# Run the exercise properly on a population where risk and value *are* related,
# so the constrained optimisation has something to find.

# %%
banner("6. SUCCESS CRITERIA -- CONSTRAINED THRESHOLD SELECTION")

realistic = generate_population(n=20_000, seed=42)
print(f"Risk-linked population: {len(realistic):,} customer-months, "
      f"{realistic['case'].sum():,} cases ({realistic['case'].mean():.2%} base rate)")

realistic_sweep = threshold_sweep(realistic, "monthly_wire_value", "case", n_thresholds=60)
CAPACITY = 2_000

best = optimise_threshold(realistic_sweep, objective="recall", max_alerts=CAPACITY)
print(f"\nOperational capacity: {CAPACITY:,} alerts per period")
print(f"Best threshold within capacity: £{best['threshold']:,.0f}")
print(f"  alerts    {int(best['alerts']):,}")
print(f"  detected  {int(best['tp']):,} of {int(best['cases']):,} cases "
      f"({best['recall']:.1%} recall)")
print(f"  precision {best['precision']:.2%}")
print(f"  effort    {best['alerts_per_true_positive']:.1f} alerts per case found")

show(realistic_sweep.loc[
        (realistic_sweep["alerts"] > CAPACITY * 0.5) & (realistic_sweep["alerts"] < CAPACITY * 2),
        ["threshold", "alerts", "tp", "precision", "recall"]],
     "\nThresholds either side of capacity -- note how recall moves with volume")

print("""
Contrast with Week 2's spec population: here precision RISES as the
threshold tightens, because value and risk are genuinely related. That
rising curve is what makes a threshold decision meaningful. Always establish
that the curve has a shape before arguing about where to sit on it.
""")

banner("END OF WEEK 2")
print("""
Carry forward into Week 3:
  * A flat precision curve is a finding, not a starting point.
  * Recall measured against SAR labels is optimistic. Always.
  * Fix the sample window before choosing the threshold, never after.
  * Capacity converts "best threshold" into a solvable question.
""")
