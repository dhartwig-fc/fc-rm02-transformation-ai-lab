# %% [markdown]
# # Week 4 -- Alert Volume and Capacity Modelling
#
# **Concept:** The statistically best threshold may be impossible operationally.
#
# **Topics**
#
# * Investigation capacity
# * Queue management
# * Alert-to-investigator ratios
# * Service-level impacts
#
# **Exercise**
#
# Assume 12 investigators, 25 alerts/day each.
#
# ```python
# capacity = 12 * 25
# ```
#
# Build alert-volume curves.
#
# **Task.** Determine the highest recall threshold without breaching capacity.
#
# **Success criteria.** Present a recommendation balancing risk, cost and
# capacity.

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
                      generate_population, optimise_threshold, show, threshold_sweep)
from tmtuning.plots import plot_alert_volume_curve, save

OUT = _ROOT / "outputs"

INVESTIGATORS = 12
ALERTS_PER_DAY_EACH = 25
WORKING_DAYS = 21
CURRENT_THRESHOLD = 50_000

# %% [markdown]
# ## 1. The capacity number

# %%
banner("1. INVESTIGATION CAPACITY")

capacity = INVESTIGATORS * ALERTS_PER_DAY_EACH
monthly_capacity = capacity * WORKING_DAYS

print(f"  capacity = {INVESTIGATORS} * {ALERTS_PER_DAY_EACH} = {capacity} alerts per day")
print(f"  monthly  = {capacity} * {WORKING_DAYS} working days = {monthly_capacity:,} alerts per month")

print("""
Before using this number, be clear about what it assumes. Every one of these
is optimistic, and each is worth challenging in a real exercise:

  * 25 alerts per investigator per day is a sustained average -- no training,
    no leave, no sickness, no QA sampling, no case write-ups, no meetings.
  * Every alert costs the same to review. In practice a corporate alert with
    forty counterparties is not one unit of the same work as a retail cash
    alert.
  * Investigators are interchangeable. Specialists, language coverage and
    approval levels all reduce effective capacity below the headline.
  * Nothing else competes for the queue -- no escalations, no requests for
    information, no remediation backlog.

Use the headline figure for the tuning arithmetic, then state the haircut you
applied and why. A capacity number quoted without its assumptions is the most
common way a tuning paper is made to balance.
""")

effective = monthly_capacity * 0.75
print(f"  A 25% haircut for the above gives {effective:,.0f} alerts/month.")
print(f"  This week uses the headline {monthly_capacity:,} as the spec sets it, but the")
print(f"  recommendation in section 7 is tested against both.")

# %% [markdown]
# ## 2. The population and the incumbent rule

# %%
banner("2. THE INCUMBENT AGAINST CAPACITY")

population = generate_population(n=250_000, seed=404, n_periods=1)
print(f"  Population        {len(population):,} customers scored this month")
print(f"  True cases        {population['case'].sum():,} ({population['case'].mean():.2%} base rate)")

current = classification_metrics(
    population["case"], apply_rule(population["monthly_wire_value"], CURRENT_THRESHOLD))

print(f"\n  Rule: monthly outbound wires > £{CURRENT_THRESHOLD:,}\n")
print(f"    Alerts generated      {current['alerts']:,}")
print(f"    Monthly capacity      {monthly_capacity:,}")
print(f"    Over capacity by      {current['alerts'] - monthly_capacity:,} "
      f"({current['alerts'] / monthly_capacity:.1f}x capacity)")
print(f"    Recall                {current['recall']:.1%}")
print(f"    Precision             {current['precision']:.2%}")

print(f"""
  This rule is generating {current['alerts'] / monthly_capacity:.1f} times what the team can work. Its {current['recall']:.0%} recall is
  therefore a paper figure: the cases it detects include a large number sitting
  in a queue nobody has reached.
""")

# %% [markdown]
# ## 3. Alert-volume curves

# %%
banner("3. ALERT-VOLUME CURVES")

sweep = threshold_sweep(population, "monthly_wire_value", "case", n_thresholds=60)
save(plot_alert_volume_curve(sweep, capacity=monthly_capacity,
                             title="Alert volume against monthly capacity"),
     OUT / "week04_alert_volume.png")
print(f"Chart written to {OUT / 'week04_alert_volume.png'}")

curve = sweep[["threshold", "alerts", "tp", "precision", "recall"]].copy()
curve["pct_of_capacity"] = curve["alerts"] / monthly_capacity
curve["alerts_per_investigator_day"] = curve["alerts"] / (INVESTIGATORS * WORKING_DAYS)
show(curve.loc[::6], "Volume curve, expressed in the units the operation uses")

print(f"""
The `alerts_per_investigator_day` column is the one to put in front of an
operations lead. "{int(curve['alerts'].iloc[0]):,} alerts a month" is abstract;
"{curve['alerts_per_investigator_day'].iloc[0]:.0f} alerts per investigator per day against a standard of {ALERTS_PER_DAY_EACH}"
is a staffing conversation with a clear answer.
""")

# %% [markdown]
# ## 4. The task: highest recall threshold within capacity

# %%
banner("4. HIGHEST RECALL THRESHOLD WITHOUT BREACHING CAPACITY")

best = optimise_threshold(sweep, objective="recall", max_alerts=monthly_capacity)
print(f"  From the grid: £{best['threshold']:,.0f}")
print(f"    alerts     {int(best['alerts']):,} ({best['alerts'] / monthly_capacity:.0%} of capacity)")
print(f"    recall     {best['recall']:.2%}")
print(f"    precision  {best['precision']:.2%}")

# %%
# A grid can only land on candidates it contains. The threshold that produces
# *exactly* capacity is a quantile lookup, not a search: take the value that
# leaves precisely `monthly_capacity` customers above it.
exact = float(np.quantile(population["monthly_wire_value"], 1 - monthly_capacity / len(population)))
exact_metrics = classification_metrics(
    population["case"], apply_rule(population["monthly_wire_value"], exact))

print(f"\n  Exact capacity threshold: £{exact:,.0f}")
print(f"    alerts     {exact_metrics['alerts']:,} ({exact_metrics['alerts'] / monthly_capacity:.1%} of capacity)")
print(f"    recall     {exact_metrics['recall']:.2%}")
print(f"    precision  {exact_metrics['precision']:.2%}")
print(f"""
  The grid search left {monthly_capacity - int(best['alerts']):,} alerts of capacity unused, which is
  {(monthly_capacity - int(best['alerts'])) / monthly_capacity:.0%} of the team's month spent on nothing. Use a grid to
  understand the shape of the curve; use the quantile to set the number.
""")

# %% [markdown]
# ## 5. The cost of the constraint

# %%
banner("5. WHAT CAPACITY COSTS IN DETECTION")

print(f"  Unconstrained, at the incumbent £{CURRENT_THRESHOLD:,}:  recall {current['recall']:.1%}")
print(f"  Constrained to {monthly_capacity:,} alerts/month:         recall {exact_metrics['recall']:.1%}")
print(f"  Detection forgone to fit the operation:    "
      f"{(current['recall'] - exact_metrics['recall']) * 100:.1f}pp "
      f"({current['tp'] - exact_metrics['tp']:,} cases)")

print("""
This is the week's concept stated as a number. The statistically preferable
threshold is not available: choosing it would not detect more risk, it would
build a backlog. But the figure above must appear in the paper, because it is
the cost of the capacity decision -- and the capacity decision belongs to the
business, not to the tuning analyst.

Presenting only the constrained answer silently accepts a resourcing
constraint on the bank's behalf. Present both.
""")

show(capacity_frontier(sweep, [monthly_capacity, int(monthly_capacity * 1.5),
                               monthly_capacity * 2, monthly_capacity * 3]),
     "What additional capacity would buy")

extra = capacity_frontier(sweep, [monthly_capacity, monthly_capacity * 2])
gain = extra.iloc[1]["tp"] - extra.iloc[0]["tp"]
print(f"\nDoubling the team (+{INVESTIGATORS} investigators) would detect "
      f"{gain:,.0f} more cases per month.")
print("Whether that is worth the cost is a business decision. Putting the")
print("number in front of them is the analyst's job.")

# %% [markdown]
# ## 6. Queue management -- what happens when you breach

# %%
banner("6. QUEUE DYNAMICS")


def simulate_queue(monthly_alerts: int, daily_capacity: int, days: int = 63) -> pd.DataFrame:
    """FIFO queue over `days` working days (3 months), alerts arriving evenly.

    Deliberately the simplest possible model: constant arrivals, constant
    service, nothing jumps the queue. It is not a forecast. It shows the
    *shape* of the failure, which is the part people consistently get wrong --
    a queue that is over capacity does not settle at a higher backlog, it grows
    without limit for as long as the breach continues.
    """
    arrivals_per_day = monthly_alerts / WORKING_DAYS
    backlog = 0.0
    rows = []
    for day in range(1, days + 1):
        backlog += arrivals_per_day
        worked = min(backlog, daily_capacity)
        backlog -= worked
        # Days for an alert arriving now to be reached, at current throughput.
        wait = backlog / daily_capacity
        rows.append({"day": day, "backlog": round(backlog), "oldest_wait_days": round(wait, 1)})
    return pd.DataFrame(rows)


for label, alerts in [("Incumbent £50k", current["alerts"]),
                      ("At capacity", exact_metrics["alerts"])]:
    queue = simulate_queue(int(alerts), capacity)
    end = queue.iloc[-1]
    print(f"  {label:<16} {int(alerts):>7,} alerts/month -> "
          f"backlog after 3 months {int(end['backlog']):>7,}, "
          f"wait {end['oldest_wait_days']:>5.1f} days")

print("""
A rule over capacity does not produce a bigger steady queue. It produces an
unbounded one: every day adds the difference between arrivals and throughput,
and the wait grows without limit until either the rule is tuned or the alerts
are written off.

The practical consequence is worse than the arithmetic suggests. Alerts are
worked FIFO, so by the time an investigator reaches a transaction it is months
old: the funds have moved, the customer relationship has changed, and a SAR
filed on it is late. Detection that arrives too late to act on is not
detection.
""")

# %% [markdown]
# ## 7. Service-level impacts

# %%
banner("7. SERVICE-LEVEL IMPACTS")

SLA_DAYS = 5
rows = []
for name, threshold in [("Incumbent", CURRENT_THRESHOLD), ("Capacity-feasible", exact)]:
    metrics = classification_metrics(
        population["case"], apply_rule(population["monthly_wire_value"], threshold))
    for label, daily in [("Headline capacity", capacity), ("With 25% haircut", capacity * 0.75)]:
        queue = simulate_queue(metrics["alerts"], daily)
        within_sla = (queue["oldest_wait_days"] <= SLA_DAYS).mean()
        rows.append({
            "threshold": threshold,
            "rule": name,
            "capacity_basis": label,
            "alerts": metrics["alerts"],
            "wait_at_3_months": queue.iloc[-1]["oldest_wait_days"],
            f"days_within_{SLA_DAYS}d_sla": within_sla,
        })
show(pd.DataFrame(rows), f"Meeting a {SLA_DAYS}-day review SLA")

print(f"""
Note what the haircut does. The capacity-feasible threshold holds the SLA on
the headline number and fails it on the realistic one -- which is exactly the
scenario where a tuning paper is signed off and the operation misses its
service levels three months later.

If capacity is the binding constraint on the recommendation, the sensitivity
of the answer to the capacity assumption is part of the evidence, not a
footnote.
""")

# %% [markdown]
# ## 8. Success criteria: a recommendation balancing risk, cost and capacity

# %%
banner("8. RECOMMENDATION")

COST_PER_ALERT = 25.0

print(f"""  PROPOSED THRESHOLD: £{exact:,.0f}  (from £{CURRENT_THRESHOLD:,})

  CAPACITY
    Alerts               {current['alerts']:,} -> {exact_metrics['alerts']:,} per month
    Against capacity     {current['alerts'] / monthly_capacity:.1f}x -> {exact_metrics['alerts'] / monthly_capacity:.2f}x
    Per investigator/day {current['alerts'] / (INVESTIGATORS * WORKING_DAYS):.0f} -> {exact_metrics['alerts'] / (INVESTIGATORS * WORKING_DAYS):.0f} (standard {ALERTS_PER_DAY_EACH})
    Backlog              unbounded -> stable

  COST
    Monthly review cost  £{current['alerts'] * COST_PER_ALERT:,.0f} -> £{exact_metrics['alerts'] * COST_PER_ALERT:,.0f}
    Effort per case      {current['alerts_per_true_positive']:.1f} -> {exact_metrics['alerts_per_true_positive']:.1f} alerts

  RISK
    Recall               {current['recall']:.1%} -> {exact_metrics['recall']:.1%}
    Cases not alerted    {current['fn']:,} -> {exact_metrics['fn']:,}
    Detection forgone    {current['tp'] - exact_metrics['tp']:,} cases per month

  THE TRADE, STATED PLAINLY

  The incumbent threshold does not detect {current['recall']:.0%} of risk. It nominally
  alerts on {current['recall']:.0%} and works {monthly_capacity / current['alerts']:.0%} of what it raises. The proposal detects
  less on paper and more in practice, because every alert it raises is
  actually reviewed, and reviewed while the information is still current.

  That argument holds only while capacity is genuinely fixed. The frontier in
  section 5 prices the alternative, and it should be presented alongside this
  recommendation rather than instead of it.
""")

answer("Capacity is an operations constraint. Why is it a tuning input at all?",
       """
Because the alternative is to let it act on the rule invisibly.

An alert that is never worked has the same detection value as an alert never
raised. If the operation can review 6,300 alerts a month, then a threshold
generating 27,000 is not a more sensitive rule -- it is the same rule plus a
queue, and the queue silently re-tunes it by working whatever reaches the top
of the list. The effective threshold becomes an accident of queue order
rather than a decision anyone made or documented.

Tuning to capacity makes that threshold explicit, deliberate and auditable.
It does not reduce detection; it stops pretending detection that was never
happening.

The honest version of the recommendation always carries both halves: this is
the best rule available at current capacity, and this is what more capacity
would buy.
""")

banner("END OF WEEK 4")
print("""
Carry forward into Week 5:
  * Capacity converts "best threshold" into a solvable question.
  * A queue over capacity grows without limit -- it never settles.
  * Quote alerts per investigator per day, not alerts per month.
  * Always price what additional capacity would buy. The resourcing decision
    is the business's to make, not yours to assume.
""")
