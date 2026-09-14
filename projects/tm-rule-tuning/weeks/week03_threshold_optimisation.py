# %% [markdown]
# # Week 3 -- Threshold Optimisation
#
# **Concept:** Tune thresholds systematically rather than relying on expert
# judgement alone.
#
# **Topics**
#
# * Sensitivity analysis
# * Threshold sweeps
# * Precision-recall trade-offs
# * Cost-based optimisation
#
# **Python Exercise**
#
# ```python
# thresholds = range(10000, 120000, 5000)
# ```
#
# For each threshold calculate precision, recall and alerts generated. Plot
# results.
#
# **Deliverable.** Produce a recommendation:
#
# * Current threshold = £50k
# * Proposed threshold = ?
#
# Supported by evidence.
#
# **Success criteria.** Document tuning rationale.

# %%
# --- path bootstrap ---
import pathlib
import sys

for _p in pathlib.Path(__file__ if "__file__" in globals() else "x").resolve().parents:
    if (_p / "src" / "tmtuning").is_dir():
        sys.path.insert(0, str(_p / "src"))
        _ROOT = _p
        break

import numpy as np
import pandas as pd

from tmtuning import (answer, banner, apply_rule, capacity_frontier, classification_metrics,
                      generate_population, marginal_yield, optimise_threshold, show,
                      split_by_period, threshold_sweep)
from tmtuning.plots import plot_precision_recall_tradeoff, plot_risk_yield_curve, save

OUT = _ROOT / "outputs"
CURRENT_THRESHOLD = 50_000
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
# ## 2. The exercise as set: a fixed sweep from £10k to £120k

# %%
banner("2. THE SPEC EXERCISE -- range(10000, 120000, 5000)")

thresholds = range(10_000, 120_000, 5_000)

rows = []
for threshold in thresholds:
    alerts = in_time["monthly_wire_value"] > threshold
    alert_count = int(alerts.sum())
    true_positives = int(in_time.loc[alerts, "case"].sum())
    rows.append({
        "threshold": threshold,
        "alerts_generated": alert_count,
        "precision": true_positives / alert_count if alert_count else np.nan,
        "recall": true_positives / in_time["case"].sum(),
    })

spec_sweep = pd.DataFrame(rows)
show(spec_sweep, "Precision, recall and alerts generated at each threshold")

# %%
# "Plot results" -- precision and recall share one axis because both are
# proportions. Never a second y-axis: two scales let the curves be slid against
# each other until they tell whichever story is wanted.
plot_frame = threshold_sweep(in_time, "monthly_wire_value", "case", thresholds=list(thresholds))
save(plot_risk_yield_curve(plot_frame, title="Precision and recall by threshold"),
     OUT / "week03_risk_yield.png")
save(plot_precision_recall_tradeoff(plot_frame, annotate_every=3,
                                    title="Precision-recall trade-off"),
     OUT / "week03_precision_recall.png")
print(f"\nCharts written to {OUT}/week03_risk_yield.png and week03_precision_recall.png")

print("""
A fixed £5k step is fine for presenting a result and poor for finding one.
It spends 22 candidates evenly across a range the data does not occupy
evenly: roughly half the customers sit below £10k, so the sweep never
examines them, while the top of the range is sampled far more finely than
its handful of customers can support. The next section builds the grid from
the data instead.
""")

# %% [markdown]
# ## 3. Sensitivity analysis
#
# Before recommending a number, ask how much the answer moves when the number
# does. A threshold sitting on a cliff is a different proposition from one on a
# plateau, even when both look identical in a results table.

# %%
banner("3. SENSITIVITY ANALYSIS")

sensitivity = []
for pct in [-20, -10, -5, 0, 5, 10, 20]:
    candidate = CURRENT_THRESHOLD * (1 + pct / 100)
    metrics = classification_metrics(
        in_time["case"], apply_rule(in_time["monthly_wire_value"], candidate))
    sensitivity.append({
        "change": f"{pct:+d}%",
        "threshold": candidate,
        "alerts": metrics["alerts"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    })

sensitivity = pd.DataFrame(sensitivity)
base_alerts = sensitivity.loc[sensitivity["change"] == "+0%", "alerts"].iloc[0]
sensitivity["alert_change"] = sensitivity["alerts"] / base_alerts - 1
show(sensitivity, f"Moving the £{CURRENT_THRESHOLD:,} threshold by +/- 20%")

swing = sensitivity["alert_change"].max() - sensitivity["alert_change"].min()
print(f"\nA +/-20% threshold move swings alert volume across a {swing:.0%} range.")
print("""
That asymmetry is the point. Alert volume responds far more sharply than
precision does, because volume follows the density of the distribution and
precision follows the much flatter risk gradient. So a threshold agreed to
the nearest round number can still be wrong by a third of the operation's
workload -- which is why the capacity work in Week 4 is not an afterthought.
""")

# %% [markdown]
# ## 4. A grid built from the data

# %%
banner("4. SYSTEMATIC THRESHOLD SWEEP")

sweep = threshold_sweep(in_time, "monthly_wire_value", "case", n_thresholds=40)
show(sweep.loc[::4, ["threshold", "alerts", "tp", "fn", "precision", "recall",
                     "alerts_per_true_positive"]],
     "Every 4th threshold from the sweep grid")

print("\nThe grid is built from the data's own QUANTILES, not evenly spaced")
print("across the range. Transaction values are heavily right-skewed: a linear")
print("grid spends most of its candidates in a tail containing almost nobody,")
print("and barely samples the dense region where the decision actually lives.")

# %% [markdown]
# ## 5. Marginal yield -- the number that should drive the decision

# %%
banner("5. MARGINAL YIELD")

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
# ## 6. Cost-based and constrained optimisation

# %%
banner("6. COST-BASED AND CONSTRAINED OPTIMISATION")

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
# ## 7. The capacity frontier

# %%
banner("7. CAPACITY FRONTIER")

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
# ## 8. Out-of-time validation -- does the threshold hold?

# %%
banner("8. OUT-OF-TIME VALIDATION")

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

# %% [markdown]
# ## 9. Deliverable: the recommendation
#
# > Current threshold = £50k
# > Proposed threshold = ?
# > Supported by evidence.

# %%
banner("9. DELIVERABLE -- TUNING RATIONALE")

current_metrics = classification_metrics(
    in_time["case"], apply_rule(in_time["monthly_wire_value"], CURRENT_THRESHOLD))
at_point = marginal.loc[(marginal["threshold"] - recommended["threshold"]).abs().idxmin()]

print(f"""  RULE:      Monthly outbound wires > threshold
  CURRENT:   £{CURRENT_THRESHOLD:,}
  PROPOSED:  £{threshold:,.0f}

  EVIDENCE

  Alert volume      {current_metrics['alerts']:,} -> {int(recommended['alerts']):,}
  Precision         {current_metrics['precision']:.2%} -> {recommended['precision']:.2%}
  Recall            {current_metrics['recall']:.2%} -> {recommended['recall']:.2%}
  Effort per case   {current_metrics['alerts_per_true_positive']:.1f} -> {recommended['alerts_per_true_positive']:.1f} alerts

  RATIONALE

  1. Constraint. Chosen to maximise recall subject to alerts <= {CAPACITY:,},
     the volume the operating model can work. The objective and the
     constraint are both stated, because both are risk decisions rather
     than technical ones.

  2. Marginal yield. At the proposed threshold the next tranche of alerts
     yields {at_point['marginal_precision']:.2%}, against a population base rate of
     {in_time['case'].mean():.2%}. Alerts beyond this point are close to being
     drawn at random.

  3. Sensitivity. A +/-20% move around the current threshold swings alert
     volume across a {swing:.0%} range, so the number needs to be set
     deliberately rather than rounded to taste.

  4. Out-of-time. The threshold holds on {out_of_time['period'].nunique()} periods held back from
     tuning (section 8), so it is not fitted to noise in the window.

  WHAT THIS RECOMMENDATION ACCEPTS

  Recall falls from {current_metrics['recall']:.1%} to {recommended['recall']:.1%}. That is a real reduction in
  detection and must be presented as such, not buried under the precision
  improvement. Week 8 puts a confidence interval around the risk it accepts.
""")

answer("Why does 'supported by evidence' mean more than showing the sweep?",
       """
Because a sweep shows what every threshold does; it does not say why you
picked one. The evidence for a recommendation is the chain that makes the
choice follow from something other than preference:

  the constraint it was chosen under, the objective it maximised subject to
  that constraint, how sensitive the answer is to the number moving, and
  whether it survived data it was not fitted to.

A paper containing a sweep and a chosen number, with no stated constraint,
has shown its working for the arithmetic and none for the decision. That is
the part independent validation actually challenges.
""")

banner("END OF WEEK 3")
print("""
Carry forward into Week 4:
  * Quantile grids, not linear ones.
  * Marginal precision, not cumulative, prices the next tranche of alerts.
  * State the objective AND the constraint. Both are risk decisions.
  * Nothing is recommended until it has held out of time.
  * Week 4 asks the harder question: can the operation actually work this
    many alerts? The statistically best threshold may be impossible.
""")
