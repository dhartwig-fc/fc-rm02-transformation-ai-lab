# %% [markdown]
# # Week 4 -- Alert Volume and Risk Yield Curves
#
# **Concept:** Turn the sweep into the two curves a decision is actually made
# from, and find the point where the trade stops being worth it.
#
# **Topics**
#
# * Alert volume curves and the "knee"
# * Risk yield curves (precision and recall against threshold)
# * The precision-recall frontier
# * Costing a threshold: investigator effort versus missed risk
#
# **Success criteria.** Produce the three charts a tuning paper needs, and
# defend a recommended operating point using them.
#
# *(Weeks 1-3 follow the course spec. Weeks 4-10 build out the remaining stated
# learning outcomes -- see LEARNING_PLAN.md for which is which.)*

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

from tmtuning import (answer, banner, capacity_frontier, generate_population, marginal_yield,
                      optimise_threshold, show, threshold_sweep)
from tmtuning.plots import (plot_alert_volume_curve, plot_precision_recall_tradeoff,
                            plot_risk_yield_curve, save)

OUT = _ROOT / "outputs"
CAPACITY = 2_000

# %%
population = generate_population(n=40_000, seed=404, n_periods=12)
sweep = threshold_sweep(population, "monthly_wire_value", "case", n_thresholds=50)

banner("1. THE THREE CURVES")
print(f"Population: {len(population):,} customer-months, {population['case'].sum():,} cases "
      f"({population['case'].mean():.2%} base rate)")
print(f"Sweep: {len(sweep)} thresholds from £{sweep['threshold'].min():,.0f} "
      f"to £{sweep['threshold'].max():,.0f}")

save(plot_alert_volume_curve(sweep, capacity=CAPACITY,
                             title="Alert volume by threshold"), OUT / "week04_alert_volume.png")
save(plot_risk_yield_curve(sweep, title="Risk yield by threshold"), OUT / "week04_risk_yield.png")
save(plot_precision_recall_tradeoff(sweep, title="Precision-recall trade-off"),
     OUT / "week04_precision_recall.png")
print(f"\nWritten to {OUT}:")
for name in ["week04_alert_volume.png", "week04_risk_yield.png", "week04_precision_recall.png"]:
    print(f"  {name}")

# %% [markdown]
# ## 2. Finding the knee of the volume curve
#
# The knee is where the curve stops falling steeply. Below it, a small
# threshold increase removes a lot of alerts. Above it, further tightening
# removes few alerts and a lot of detection.

# %%
banner("2. THE KNEE OF THE VOLUME CURVE")

curve = sweep[["threshold", "alerts", "tp", "precision", "recall"]].copy()
# Elasticity: the % change in alert volume for a 1% change in threshold. A
# log-log slope, so it is scale-free and comparable across rules of any size.
curve["elasticity"] = (np.gradient(np.log(curve["alerts"].clip(lower=1)))
                       / np.gradient(np.log(curve["threshold"])))
show(curve.loc[::5], "Volume elasticity along the curve")

knee = curve.loc[curve["elasticity"].abs().idxmax()]
print(f"\nSteepest response at £{knee['threshold']:,.0f} "
      f"(elasticity {knee['elasticity']:.2f}: a 1% threshold rise removes "
      f"{abs(knee['elasticity']):.2f}% of alerts)")
print("""
Elasticity matters because a threshold sitting on a very steep part of the
curve is FRAGILE. A modest shift in customer behaviour -- inflation, a new
product, a portfolio acquisition -- moves alert volume sharply, and the
operation cannot absorb it. Two thresholds with identical precision today can
have very different volume risk tomorrow. Prefer the flatter one where the
detection cost of doing so is small, and say in the paper that you did.
""")

# %% [markdown]
# ## 3. Costing the trade-off
#
# Both arms of the trade have a cost. Making them commensurable is the only way
# to argue about the operating point without hand-waving.

# %%
banner("3. COSTING THE OPERATING POINT")

COST_PER_ALERT = 25.0           # fully loaded investigator cost per alert reviewed
COST_PER_MISSED_CASE = 5_000.0  # placeholder for the cost of undetected risk

costed = sweep[["threshold", "alerts", "tp", "fn", "precision", "recall"]].copy()
costed["investigation_cost"] = costed["alerts"] * COST_PER_ALERT
costed["missed_risk_cost"] = costed["fn"] * COST_PER_MISSED_CASE
costed["total_cost"] = costed["investigation_cost"] + costed["missed_risk_cost"]
show(costed.loc[::5], f"Cost model at £{COST_PER_ALERT:,.0f}/alert and "
                      f"£{COST_PER_MISSED_CASE:,.0f}/missed case")

cheapest = costed.loc[costed["total_cost"].idxmin()]
print(f"\nMinimum modelled total cost at £{cheapest['threshold']:,.0f} "
      f"(£{cheapest['total_cost']:,.0f})")

answer("Should the cost minimum be the recommended threshold?",
       """
No -- not on its own, and the reason is the second input.

Cost per alert is knowable: it is a headcount number the operation can
evidence. Cost per missed case is not. It stands in for regulatory
censure, enforcement, remediation and the underlying harm, none of which
are linear in the number of cases and none of which the bank gets to price.
Change that single assumption from £5,000 to £50,000 and the "optimal"
threshold moves a long way.

So use the cost model to show the SHAPE -- that total cost is flat across a
broad range, or that it rises sharply past a point -- and use the capacity
constraint plus risk appetite to pick within it. A recommendation resting on
an unevidenced cost-per-missed-case is resting on an assumption its author
made up, and a validator will say so.
""")

# %%
sensitivity = []
for cost in [1_000, 5_000, 25_000, 100_000]:
    trial = costed["alerts"] * COST_PER_ALERT + costed["fn"] * cost
    row = costed.loc[trial.idxmin()]
    sensitivity.append({"cost_per_missed_case": cost, "optimal_threshold": row["threshold"],
                        "alerts": int(row["alerts"]), "recall": row["recall"]})
show(pd.DataFrame(sensitivity), "Sensitivity: the 'optimum' is an artefact of one assumption")

# %% [markdown]
# ## 4. Recommending an operating point

# %%
banner("4. RECOMMENDED OPERATING POINT")

recommended = optimise_threshold(sweep, objective="recall", max_alerts=CAPACITY)
marginal = marginal_yield(sweep)
at_point = marginal.loc[(marginal["threshold"] - recommended["threshold"]).abs().idxmin()]

print(f"  Threshold             £{recommended['threshold']:,.0f}")
print(f"  Alerts                {int(recommended['alerts']):,} (capacity {CAPACITY:,})")
print(f"  Capacity utilisation  {recommended['alerts'] / CAPACITY:.1%}")
print(f"  Cases detected        {int(recommended['tp']):,} of {int(recommended['cases']):,}")
print(f"  Precision             {recommended['precision']:.2%}")
print(f"  Recall                {recommended['recall']:.2%}")
print(f"  Effort per case       {recommended['alerts_per_true_positive']:.1f} alerts")
print(f"  Marginal precision    {at_point['marginal_precision']:.2%} "
      f"(vs {population['case'].mean():.2%} base rate)")
print(f"  Monthly cost          £{recommended['alerts'] * COST_PER_ALERT:,.0f}")

show(capacity_frontier(sweep, [1_000, 1_500, 2_000, 2_500, 3_000]),
     "\nWhat a different budget would buy, for the resourcing conversation")

banner("END OF WEEK 4")
print("""
Carry forward into Week 5:
  * The knee tells you where volume becomes fragile, not just where it falls.
  * Any cost model is only as good as its least evidenced input. Show the
    sensitivity, then choose on capacity and appetite.
  * Every chart in this week's outputs belongs in the tuning paper.
""")
