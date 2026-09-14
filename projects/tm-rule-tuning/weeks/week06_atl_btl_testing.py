# %% [markdown]
# # Week 6 -- Above and Below-the-Line Testing
#
# **Concept:** Every metric so far measures risk the rule already found.
# Below-the-line testing measures the risk it did not -- the only part of the
# picture production data cannot give you.
#
# **Topics**
#
# * Above-the-line (ATL) testing -- are the alerts productive?
# * Below-the-line (BTL) testing -- what is being missed?
# * Sample size planning
# * Wilson confidence intervals, and why not the textbook interval
# * Extrapolating a sample rate to the full below-the-line population
#
# **Success criteria.** State the missed-risk estimate with a confidence
# interval, and size the sample *before* drawing it.

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

from tmtuning import (answer, banner, apply_rule, atl_btl_sample, btl_test, classification_metrics,
                      generate_population, required_sample_size, show, wilson_interval)

THRESHOLD = 50_000

# %%
banner("1. THE LINE")

population = generate_population(n=40_000, seed=606, n_periods=12)
alerts = apply_rule(population["monthly_wire_value"], THRESHOLD)

above = int(alerts.sum())
below = int((~alerts).sum())
print(f"Rule: monthly outbound wires > £{THRESHOLD:,}\n")
print(f"  Above the line : {above:>7,} records  ({above / len(population):.2%} of population)")
print(f"  Below the line : {below:>7,} records  ({below / len(population):.2%} of population)")
print("""
Production data tells you everything about the top row and nothing about the
bottom one. Below the line nobody looked, so nobody filed, so the records
carry no label -- and a rule evaluated only on labelled data scores itself on
a population it selected. Below-the-line testing breaks that circularity by
sampling records the rule deliberately ignored.
""")

# %% [markdown]
# ## 2. Size the sample before drawing it

# %%
banner("2. SAMPLE SIZE PLANNING")

plan = []
for expected in [0.01, 0.02, 0.05]:
    for margin in [0.02, 0.01, 0.005]:
        plan.append({
            "expected_rate": expected,
            "margin_of_error": margin,
            "sample_size_95": required_sample_size(expected, margin, 0.95),
            "sample_size_99": required_sample_size(expected, margin, 0.99),
        })
show(pd.DataFrame(plan), "Records to review for a given precision on the estimate")

print("""
Read the trade-off: halving the margin of error roughly quadruples the sample.
That is the arithmetic behind "we reviewed 100 files" being an inadequate BTL
test -- at a 2% expected rate, 100 records gives a 95% interval of roughly
0% to 7%, which cannot distinguish "nothing is being missed" from "three and
a half times the acceptable rate is being missed".

Size the sample from the precision the decision needs, then negotiate the
review effort. Doing it the other way round produces a number that cannot
answer the question it was collected for.
""")

# %% [markdown]
# ## 3. Above-the-line testing

# %%
banner("3. ABOVE-THE-LINE TESTING")

samples = atl_btl_sample(population, alerts, n_above=200, n_below=1_500, seed=606)
atl = samples["above"]
atl_productive = int(atl["case"].sum())
atl_lo, atl_hi = wilson_interval(atl_productive, len(atl))

print(f"  Sampled        {len(atl):,} alerts")
print(f"  Productive     {atl_productive:,}")
print(f"  Observed rate  {atl_productive / len(atl):.2%}")
print(f"  95% interval   {atl_lo:.2%} to {atl_hi:.2%}")
print(f"  Full ATL rate  {population.loc[alerts, 'case'].mean():.2%}  (the true value, known here only because the data is synthetic)")

print("""
ATL testing confirms the alerts a rule raises are worth raising. It is the
easier half: the records have been investigated, so the labels exist. Its main
use in a threshold change is to show that the alerts ADDED by a loosened
threshold are productive, not just that the existing ones are.
""")

# %% [markdown]
# ## 4. Below-the-line testing -- the one that matters

# %%
banner("4. BELOW-THE-LINE TESTING")

result = btl_test(population, alerts, n_below=1_500, seed=606)

print(f"  Below-the-line population   {result['below_the_line_population']:,}")
print(f"  Sampled                     {result['sampled']:,}")
print(f"  True cases found in sample  {result['productive_in_sample']:,}")
print(f"  Observed rate               {result['observed_rate']:.2%}")
print(f"  95% Wilson interval         {result['rate_lower']:.2%} to {result['rate_upper']:.2%}")
print()
print(f"  Estimated cases missed      {result['estimated_missed_cases']:,.0f}")
print(f"  Upper bound                 {result['estimated_missed_cases_upper']:,.0f}")
print(f"  Actual cases missed         {result['actual_missed_cases']:,}  (synthetic data only)")

inside = result["rate_lower"] <= result["actual_missed_cases"] / result["below_the_line_population"] <= result["rate_upper"]
print(f"\n  True rate falls inside the interval: {inside}")

print("""
Quote the UPPER bound in the paper, not the point estimate. A reviewer's next
question after "roughly 600 missed cases" is always "could it be worse than
that?" -- the upper bound answers it in advance, and it is the figure risk
appetite should be tested against.
""")

# %% [markdown]
# ## 5. Why Wilson, not the textbook interval

# %%
banner("5. WHY WILSON")

print("The textbook (Wald) interval is  p +/- z * sqrt(p(1-p)/n).")
print("Watch what it does when a sample turns up no cases at all:\n")

comparison = []
for successes, trials in [(0, 100), (0, 500), (1, 300), (5, 300), (30, 300)]:
    p = successes / trials
    wald_half = 1.96 * np.sqrt(p * (1 - p) / trials)
    lo, hi = wilson_interval(successes, trials)
    comparison.append({
        "successes": successes, "trials": trials, "observed": p,
        "wald_low": max(p - wald_half, 0), "wald_high": p + wald_half,
        "wilson_low": lo, "wilson_high": hi,
    })
show(pd.DataFrame(comparison), "Wald versus Wilson")

answer("Why does the zero-case row matter so much?",
       """
Because it is the row a BTL test most often produces, and the row a tuning
paper most wants to lean on.

With 0 cases in 100 records, Wald returns the interval [0%, 0%] -- zero
width. Taken at face value that says the below-the-line population contains
no risk, with certainty, on the basis of 100 files. Wilson returns
[0%, 3.7%], which is the honest reading: the sample is consistent with
anything up to roughly 3.7%, and across a below-the-line population of
30,000 that is up to about 1,100 missed cases.

Same data, same review effort. One interval supports a threshold increase
and the other does not.
""")

# %% [markdown]
# ## 6. Testing a proposed threshold change

# %%
banner("6. BTL EVIDENCE FOR A THRESHOLD CHANGE")

rows = []
for candidate in [25_000, 50_000, 75_000, 100_000]:
    flags = apply_rule(population["monthly_wire_value"], candidate)
    metrics = classification_metrics(population["case"], flags)
    btl = btl_test(population, flags, n_below=1_500, seed=606)
    rows.append({
        "threshold": candidate,
        "alerts": metrics["alerts"],
        "precision": metrics["precision"],
        "btl_rate": btl["observed_rate"],
        "btl_rate_upper": btl["rate_upper"],
        "missed_estimate": btl["estimated_missed_cases"],
        "missed_upper": btl["estimated_missed_cases_upper"],
    })
show(pd.DataFrame(rows), "Missed-risk estimate at each candidate threshold")

print("""
This is the table that turns a threshold increase from an efficiency claim
into a risk decision. "We can cut alerts by 40%" and "we would be accepting
up to N additional missed cases" are the same proposal described from the two
ends. A tuning paper that only contains the first is incomplete, and an
independent validator will send it back.

One caveat to state explicitly: the BTL rate here is measured against the same
label set as everything else. Below the line those labels are sparsest, so
even this estimate is a lower bound on true missed risk. It narrows the
uncertainty; it does not eliminate it.
""")

banner("END OF WEEK 6")
print("""
Carry forward into Week 7:
  * Size the sample from the precision the decision needs. Then negotiate.
  * Wilson, always. Wald's zero-width interval at zero cases is a trap.
  * Report the upper bound, not the point estimate.
  * Threshold changes need a missed-risk number, not just a volume number.
""")
