# %% [markdown]
# # Week 3 -- Threshold Optimisation
#
# **Concept:** Tune thresholds systematically rather than relying on expert
# judgement alone.
#
# **Topics**
#
# * Systematic threshold sweeps versus judgement-led adjustment
# * Marginal yield -- pricing the alerts you are about to add
# * Constrained optimisation against operational capacity
# * Overfitting, and out-of-time validation
#
# **Success criteria.** Recommend a threshold, state the constraint it was
# chosen under, and show that it holds on data it was not tuned on.

# %%
# --- path bootstrap ---
import pathlib
import sys

for _p in pathlib.Path(__file__ if "__file__" in globals() else "x").resolve().parents:
    if (_p / "src" / "tmtuning").is_dir():
        sys.path.insert(0, str(_p / "src"))
        break

import pandas as pd

from tmtuning import (answer, banner, apply_rule, capacity_frontier, classification_metrics,
                      generate_population, marginal_yield, optimise_threshold, show,
                      split_by_period, threshold_sweep)

CAPACITY = 1_500

# %% [markdown]
# ## 1. Why judgement alone is not enough

# %%
banner("1. JUDGEMENT VERSUS EVIDENCE")

print("""
Expert judgement is good at the things evidence is bad at: knowing that a
typology exists, that a product is being abused, that a threshold sits on a
round number criminals structure beneath. It is unreliable at one specific
thing -- estimating how a change in a number will move alert volume and
detection, because both depend on the shape of a distribution nobody can
hold in their head.

So the division of labour is: judgement sets the candidate range and vetoes
results that make no typological sense. Evidence picks the point within it.
""")

# %%
population = generate_population(n=40_000, seed=2024, n_periods=12)
in_time, out_of_time = split_by_period(population, holdout_periods=3)

print(f"Full population : {len(population):,} customer-months, "
      f"{population['case'].sum():,} cases ({population['case'].mean():.2%})")
print(f"In-time (tune)  : {len(in_time):,} rows, periods "
      f"{in_time['period'].min()} to {in_time['period'].max()}")
print(f"Out-of-time     : {len(out_of_time):,} rows, periods "
      f"{out_of_time['period'].min()} to {out_of_time['period'].max()}")
print("\nThe split is on TIME, not at random. A random split lets the tuning")
print("sample and the validation sample share the same months, so a threshold")
print("that only works under one quarter's conditions still validates cleanly.")

# %% [markdown]
# ## 2. The systematic sweep

# %%
banner("2. SYSTEMATIC THRESHOLD SWEEP")

sweep = threshold_sweep(in_time, "monthly_wire_value", "case", n_thresholds=40)
show(sweep.loc[::4, ["threshold", "alerts", "tp", "fn", "precision", "recall",
                     "alerts_per_true_positive"]],
     "Every 4th threshold from the sweep grid")

print("\nThe grid is built from the data's own QUANTILES, not evenly spaced")
print("across the range. Transaction values are heavily right-skewed: a linear")
print("grid spends most of its candidates in a tail containing almost nobody,")
print("and barely samples the dense region where the decision actually lives.")

# %% [markdown]
# ## 3. Marginal yield -- the number that should drive the decision

# %%
banner("3. MARGINAL YIELD")

marginal = marginal_yield(sweep)
show(marginal.loc[::4, ["threshold", "alerts", "precision", "extra_alerts", "extra_tp",
                        "marginal_precision", "alerts_per_extra_tp"]],
     "Cost of loosening the threshold one step at a time (tightest first)")

base_rate = in_time["case"].mean()
print(f"\nPopulation base rate: {base_rate:.2%}")
print("""
`precision` is cumulative -- it averages in the highly productive top of the
distribution. `marginal_precision` prices only the alerts you are about to
ADD. It is always the worse number, and it is the honest one.

Read it like this: when marginal precision falls to roughly the population
base rate, the next tranche of alerts is no better than picking customers at
random. There is no detection argument for going below that point, only a
coverage-optics one.
""")

near_random = marginal[marginal["marginal_precision"] <= base_rate * 1.2].dropna(subset=["marginal_precision"])
if not near_random.empty:
    point = near_random.iloc[0]
    print(f"Marginal precision first falls within 20% of the base rate at "
          f"£{point['threshold']:,.0f}")
    print(f"  ({point['marginal_precision']:.2%} marginal vs {base_rate:.2%} base rate)")

# %% [markdown]
# ## 4. Constrained optimisation

# %%
banner("4. CONSTRAINED OPTIMISATION")

recommended = optimise_threshold(sweep, objective="recall", max_alerts=CAPACITY)
print(f"Constraint: alerts <= {CAPACITY:,} per the operating model")
print(f"Objective : maximise recall\n")
print(f"  Recommended threshold  £{recommended['threshold']:,.0f}")
print(f"  Alerts                 {int(recommended['alerts']):,}")
print(f"  Cases detected         {int(recommended['tp']):,} of {int(recommended['cases']):,}")
print(f"  Precision              {recommended['precision']:.2%}")
print(f"  Recall                 {recommended['recall']:.2%}")

# %%
print("\nThe same sweep under different objectives -- note they disagree:\n")
objectives = []
for objective in ["recall", "precision", "f1"]:
    row = optimise_threshold(sweep, objective=objective, max_alerts=CAPACITY)
    objectives.append({
        "objective": objective, "threshold": row["threshold"], "alerts": int(row["alerts"]),
        "tp": int(row["tp"]), "precision": row["precision"], "recall": row["recall"],
    })
show(pd.DataFrame(objectives), "Objective choice is a risk appetite decision, not a technical one")

answer("Which objective should a tuning paper use?",
       """
Recall, subject to a capacity constraint -- and say so explicitly.

Maximising precision alone drives the threshold up until the rule alerts on
almost nothing at excellent yield, which is how a rule quietly stops
detecting anything. Maximising F1 hides a value judgement inside a formula:
it prices one missed case exactly equal to one wasted investigation, which
no financial crime function actually believes and none would defend in
writing.

Stating it as "maximise detection subject to alerts <= capacity" puts the
trade-off where it belongs -- visible, owned by the risk owner, and arguable.
""")

# %% [markdown]
# ## 5. The capacity frontier

# %%
banner("5. CAPACITY FRONTIER")

frontier = capacity_frontier(sweep, capacities=[500, 1_000, 1_500, 2_000, 3_000, 4_000])
frontier["cases_per_1k_alerts"] = frontier["tp"] / (frontier["alerts"] / 1_000)
show(frontier, "What each alert budget buys in detection")

print("""
This is the table for the conversation about resourcing, because it prices
detection in headcount. Note the diminishing returns: each additional
thousand alerts buys fewer additional cases than the last. That curve, not a
target precision figure, is the argument for or against funding another
investigator.
""")

# %% [markdown]
# ## 6. Out-of-time validation -- does the threshold hold?

# %%
banner("6. OUT-OF-TIME VALIDATION")

threshold = float(recommended["threshold"])
comparison = pd.DataFrame([
    {"sample": "In-time (tuned on)",
     **classification_metrics(in_time["case"], apply_rule(in_time["monthly_wire_value"], threshold))},
    {"sample": "Out-of-time (held back)",
     **classification_metrics(out_of_time["case"], apply_rule(out_of_time["monthly_wire_value"], threshold))},
])
show(comparison[["sample", "alerts", "tp", "precision", "recall", "alert_rate",
                 "alerts_per_true_positive"]],
     f"Threshold £{threshold:,.0f} applied to both samples")

# Signed as (out-of-time - in-time), so a negative number reads as a drop.
change = comparison.loc[1, "precision"] - comparison.loc[0, "precision"]
direction = "lower" if change < 0 else "higher"
print(f"\nPrecision out of time is {abs(change) * 100:.2f}pp {direction} "
      f"({comparison.loc[0, 'precision']:.2%} -> {comparison.loc[1, 'precision']:.2%}).")
print("""
Compare RATES, never raw counts, across the two samples -- the holdout is a
smaller number of periods, so its alert count is lower by construction and
that tells you nothing.

A large precision drop out of time means the threshold was fitted to noise in
the tuning window. The usual causes are a grid so fine that it can chase
individual records, and a tuning window too short to average out seasonality.
A threshold that cannot survive three months it has not seen will not survive
production.
""")

banner("END OF WEEK 3")
print("""
Carry forward into Week 4:
  * Quantile grids, not linear ones.
  * Marginal precision, not cumulative, prices the next tranche of alerts.
  * State the objective AND the constraint. Both are risk decisions.
  * Nothing is recommended until it has held out of time.
""")
