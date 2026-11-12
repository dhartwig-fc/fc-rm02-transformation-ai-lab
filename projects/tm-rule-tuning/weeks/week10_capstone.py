# %% [markdown]
# # Week 10 -- Capstone Backtest and Calibration Project
#
# **Objective.** Complete an end-to-end transaction-monitoring tuning
# engagement.
#
# ## Capstone scenario
#
# Existing rule:
#
# > **Cash deposits > £10k in 30 days**
#
# Population: 100,000 customers, with fields for Customer ID, Segment, Country
# Risk, Cash Deposits, Wire Activity, Velocity, Alert Outcome and Case Outcome.
#
# Constraints:
#
# * Investigation capacity fixed
# * Senior management expects volume reduction
# * Missed-risk tolerance limited
#
# ## Required analysis
#
# 1. Baseline assessment -- alert volume, precision, recall, FPR
# 2. Threshold sweep -- £10k, £15k, £20k, £25k, £30k
# 3. Segment calibration -- global, retail and corporate thresholds
# 4. Challenger design -- cash threshold + velocity condition
# 5. Stability testing -- Q1, Q2, Q3, Q4
# 6. Recommendation -- proposed calibration, impact and risks

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

from tmtuning import (answer, banner, apply_rule, classification_metrics, generate_population,
                      mini_tuning_paper, save_paper, show, threshold_sweep, TuningDecisionLog)
from tmtuning.plots import plot_alert_volume_curve, plot_stability_chart, save
from tmtuning.stability import stability_report

OUT = _ROOT / "outputs"

CURRENT_THRESHOLD = 10_000        # cash deposits > £10k in 30 days
SWEEP_THRESHOLDS = [10_000, 15_000, 20_000, 25_000, 30_000]

# Constraint 1: investigation capacity is fixed. The team of 12 works
# 12 * 25 * 21 = 6,300 alerts a month across the whole rule estate; TM-021's
# agreed share of that queue is 1,500.
TEAM_CAPACITY = 12 * 25 * 21
CAPACITY = 1_500

log = TuningDecisionLog(rule="TM-021 Cash Deposit Structuring", author="Dan Hartwig")

# %% [markdown]
# ## Population

# %%
banner("POPULATION")

population = generate_population(n=100_000, seed=1010, n_periods=12,
                                 start_period="2025-01", drift_strength=1.2)
population["quarter"] = "Q" + (population["period_index"] // 3 + 1).astype(str)

# The spec's two-segment view: personal banking versus everything commercial.
population["segment_group"] = np.where(population["segment"] == "RETAIL", "Retail", "Corporate")

# Alert Outcome is the disposition the INCUMBENT rule produced. It is derived
# from the incumbent, not given -- which is the whole point of section 1's
# warning: the outcome field only exists for records the live rule alerted on.
incumbent_flags = np.asarray(apply_rule(population["monthly_cash_deposits"], CURRENT_THRESHOLD))
population["alert_outcome"] = np.select(
    [~incumbent_flags, incumbent_flags & (population["case"] == 1)],
    ["NO ALERT", "SAR FILED"], default="CLOSED - NO ACTION")
population["case_outcome"] = np.where(population["case"] == 1, "CASE", "NO CASE")

capstone = population.rename(columns={
    "customer_id": "Customer ID", "segment_group": "Segment",
    "jurisdiction_risk": "Country Risk", "monthly_cash_deposits": "Cash Deposits",
    "monthly_wire_value": "Wire Activity", "velocity": "Velocity",
    "alert_outcome": "Alert Outcome", "case_outcome": "Case Outcome",
})
show(capstone[["Customer ID", "Segment", "Country Risk", "Cash Deposits", "Wire Activity",
               "Velocity", "Alert Outcome", "Case Outcome"]].head(6),
     "The eight specified fields")

print(f"\n  Records          {len(population):,} customer-months across 4 quarters")
print(f"  True cases       {population['case'].sum():,} ({population['case'].mean():.2%} base rate)")
print(f"  Team capacity    {TEAM_CAPACITY:,} alerts/month across the estate")
print(f"  TM-021 allocated {CAPACITY:,} alerts/month ({CAPACITY / TEAM_CAPACITY:.0%} of the queue)")

show(population.groupby("Segment" if "Segment" in population else "segment_group").agg(
        records=("case", "size"), cases=("case", "sum"), prevalence=("case", "mean"),
        median_cash=("monthly_cash_deposits", "median")).reset_index(),
     "\nSegment profile")

print("""
  A note on Alert Outcome before using it. It is a function of the incumbent
  rule: every record below £10,000 reads "NO ALERT", not because it was
  reviewed and cleared but because nobody looked. Treating "NO ALERT" as
  evidence of no risk is the circularity Week 1 warned about, and it is the
  single easiest way to make a capstone answer look better than it is.
""")

# %% [markdown]
# ## 1. Baseline assessment

# %%
banner("1. BASELINE ASSESSMENT")

baseline = classification_metrics(population["case"], incumbent_flags)
print(f"  Rule: cash deposits > £{CURRENT_THRESHOLD:,} in 30 days\n")
print(f"    Alert volume   {baseline['alerts']:,} over 12 periods "
      f"({baseline['alerts'] / 12:,.0f} per month)")
print(f"    Precision      {baseline['precision']:.2%}")
print(f"    Recall         {baseline['recall']:.2%}")
print(f"    FPR            {baseline['fpr']:.2%}")
print(f"\n    Against the {CAPACITY:,}/month allocation: "
      f"{baseline['alerts'] / 12 / CAPACITY:.0%} of capacity "
      f"({baseline['alerts'] / 12 - CAPACITY:+,.0f} alerts/month)")

log.record("1. Baseline", f"cash > £{CURRENT_THRESHOLD:,} (incumbent)", baseline,
           operational_implication=f"{baseline['alerts'] / 12:,.0f} alerts/month against a "
                                   f"{CAPACITY:,} allocation -- {baseline['alerts'] / 12 / CAPACITY:.0%} of capacity.",
           assumptions="Case label = SAR/STR filed within 90 days of the alert period.",
           outcome="for information",
           rationale="Baseline as currently running. Breaches capacity; volume reduction required.")

print("""
  FPR is reported because the spec asks for it, and it is the metric most often
  misread in this pack. 25.6% sounds catastrophic beside a 7.7% precision, but
  the two answer different questions: FPR is unproductive alerts as a share of
  all NON-cases, and non-cases are 97% of the book. At AML base rates FPR is
  almost a restatement of the alert rate, and it moves very little whatever the
  threshold does. Precision is what tells you about investigator experience.
""")

# %% [markdown]
# ## 2. Threshold sweep

# %%
banner("2. THRESHOLD SWEEP")

sweep = threshold_sweep(population, "monthly_cash_deposits", "case",
                        thresholds=SWEEP_THRESHOLDS)
sweep["alerts_per_month"] = sweep["alerts"] / 12
sweep["pct_of_capacity"] = sweep["alerts_per_month"] / CAPACITY
show(sweep[["threshold", "alerts", "alerts_per_month", "pct_of_capacity",
            "precision", "recall", "fpr"]],
     "The five thresholds the spec asks for")

for _, row in sweep.iterrows():
    fits = row["alerts_per_month"] <= CAPACITY
    log.record("2. Threshold sweep", f"cash > £{row['threshold']:,.0f}", row.to_dict(),
               operational_implication=f"{row['alerts_per_month']:,.0f} alerts/month "
                                       f"({row['pct_of_capacity']:.0%} of allocation).",
               outcome="carried forward" if fits else "rejected",
               rationale=("Fits the allocation." if fits
                          else "Exceeds the fixed investigation capacity."))

feasible = sweep[sweep["alerts_per_month"] <= CAPACITY]
best_global = feasible.sort_values("recall", ascending=False).iloc[0]
print(f"\n  Highest-recall global threshold within capacity: £{best_global['threshold']:,.0f}")
print(f"    {best_global['alerts_per_month']:,.0f} alerts/month, "
      f"recall {best_global['recall']:.2%}, precision {best_global['precision']:.2%}")
print(f"    Recall given up against the incumbent: "
      f"{(baseline['recall'] - best_global['recall']) * 100:.1f}pp")

full_sweep = threshold_sweep(population, "monthly_cash_deposits", "case", n_thresholds=60)
save(plot_alert_volume_curve(full_sweep, capacity=CAPACITY * 12,
                             title="TM-021 alert volume against allocated capacity"),
     OUT / "week10_alert_volume.png")

# %% [markdown]
# ## 3. Segment calibration

# %%
banner("3. SEGMENT CALIBRATION")

retail = population[population["segment_group"] == "Retail"]
corporate = population[population["segment_group"] == "Corporate"]
print(f"  Retail    {len(retail):>7,} records, median cash deposits "
      f"£{retail['monthly_cash_deposits'].median():>9,.0f}, prevalence {retail['case'].mean():.2%}")
print(f"  Corporate {len(corporate):>7,} records, median cash deposits "
      f"£{corporate['monthly_cash_deposits'].median():>9,.0f}, prevalence {corporate['case'].mean():.2%}")


def evaluate_segmented(retail_cut: float, corporate_cut: float) -> dict:
    """Score a two-threshold rule across the whole population."""
    flags = np.where(population["segment_group"] == "Retail",
                     population["monthly_cash_deposits"] > retail_cut,
                     population["monthly_cash_deposits"] > corporate_cut)
    return classification_metrics(population["case"], flags)


rows = [{"option": f"Global £{best_global['threshold']:,.0f}",
         **classification_metrics(population["case"],
                                  apply_rule(population["monthly_cash_deposits"],
                                             best_global["threshold"]))}]
for retail_cut, corporate_cut in [(10_000, 25_000), (15_000, 35_000), (20_000, 40_000)]:
    rows.append({"option": f"Retail £{retail_cut:,} / Corporate £{corporate_cut:,}",
                 **evaluate_segmented(retail_cut, corporate_cut)})

segment_options = pd.DataFrame(rows)
segment_options["alerts_per_month"] = segment_options["alerts"] / 12
show(segment_options[["option", "alerts", "alerts_per_month", "precision", "recall", "fpr"]],
     "Global versus segment-specific thresholds")

for _, row in segment_options.iterrows():
    fits = row["alerts_per_month"] <= CAPACITY
    log.record("3. Segment calibration", row["option"], row.to_dict(),
               operational_implication=f"{row['alerts_per_month']:,.0f} alerts/month.",
               assumptions="Segment assignment is accurate and stable; it becomes an AML control.",
               outcome="carried forward" if fits else "rejected",
               rationale=("Within capacity." if fits else "Exceeds capacity."))

print("""
  Compare these at capacity, not at face value. A segmented option firing more
  alerts than the global one will find more cases, and that is arithmetic
  rather than calibration. The question is whether it finds more for the same
  investigator effort.
""")

# %% [markdown]
# ## 4. Challenger design
#
# Cash threshold **+** velocity condition.

# %%
banner("4. CHALLENGER DESIGN")

velocity_profile = pd.DataFrame([
    {"condition": f"velocity > {v}",
     "keeps": (population["velocity"] > v).mean(),
     "case_rate_kept": population.loc[population["velocity"] > v, "case"].mean(),
     "case_rate_excluded": population.loc[population["velocity"] <= v, "case"].mean()}
    for v in (0, 1, 2, 3)
])
velocity_profile["lift"] = velocity_profile["case_rate_kept"] / velocity_profile["case_rate_excluded"]
show(velocity_profile, "What each velocity floor does on its own")

print("""
  Note what `velocity > 0` does, because it looks trivial and is not. It keeps
  any customer with at least one outbound transaction, excluding those with
  none -- and that excluded group has a far lower case rate.

  There is a typology behind that. Cash deposited and left sitting is not
  layering; cash deposited and then moved is. A velocity floor of zero is
  really the condition "the money went somewhere afterwards", which is exactly
  what the cash rule alone cannot see.
""")


def challenger_flags(df: pd.DataFrame, retail_cut: float, corporate_cut: float,
                     velocity_min: int | None) -> np.ndarray:
    """Segment cash thresholds AND an optional velocity condition.

    ``velocity_min=None`` means no velocity condition at all. ``0`` is a real
    condition -- "at least one outbound transaction" -- not the absence of one.
    """
    cash = np.where(df["segment_group"] == "Retail",
                    df["monthly_cash_deposits"] > retail_cut,
                    df["monthly_cash_deposits"] > corporate_cut)
    return cash if velocity_min is None else cash & (df["velocity"] > velocity_min)


def evaluate_challenger(retail_cut: float, corporate_cut: float,
                        velocity_min: int | None, df: pd.DataFrame | None = None) -> dict:
    """Score a challenger design on ``df`` (the full population by default)."""
    df = population if df is None else df
    return classification_metrics(df["case"], challenger_flags(df, retail_cut, corporate_cut, velocity_min))


def describe(retail_cut: float, corporate_cut: float, velocity_min: int | None) -> str:
    base = (f"Cash > £{retail_cut:,} (all segments)" if retail_cut == corporate_cut
            else f"Retail £{retail_cut:,} / Corporate £{corporate_cut:,}")
    if velocity_min is None:
        return base + " (no velocity condition)"
    return base + f" AND velocity > {velocity_min}"


rows = []
for retail_cut, corporate_cut, velocity_min in [
    (15_000, 35_000, None), (15_000, 35_000, 0), (15_000, 35_000, 2), (15_000, 35_000, 3),
]:
    rows.append({"option": describe(retail_cut, corporate_cut, velocity_min),
                 **evaluate_challenger(retail_cut, corporate_cut, velocity_min)})

challengers = pd.DataFrame(rows)
challengers["alerts_per_month"] = challengers["alerts"] / 12
show(challengers[["option", "alerts", "alerts_per_month", "precision", "recall", "fpr"]],
     "Challenger designs")

for _, row in challengers.iterrows():
    fits = row["alerts_per_month"] <= CAPACITY
    log.record("4. Challenger design", row["option"], row.to_dict(),
               operational_implication=f"{row['alerts_per_month']:,.0f} alerts/month.",
               assumptions="Velocity is populated for every customer in the production engine.",
               outcome="carried forward" if fits else "rejected",
               rationale=("Within capacity." if fits else "Exceeds capacity."))

print("""
  The velocity condition is an AND, so it can only remove alerts. It buys
  precision and pays in recall, and the size of that trade is the whole design
  decision -- a velocity floor set too high turns a monitoring rule into a
  rule that detects only customers who were never trying to hide.

  One implementation risk to carry into the recommendation: a null velocity
  fails the AND closed. The customer is then not monitored by this rule at all,
  silently, and the backtest above cannot see it because the synthetic field is
  always populated.
""")

# %% [markdown]
# ## 5. Stability testing across Q1-Q4

# %%
banner("5. STABILITY TESTING")

SPEC_RETAIL, SPEC_CORPORATE, SPEC_VELOCITY = 15_000, 35_000, 3

candidates = {
    f"Incumbent (cash > £{CURRENT_THRESHOLD:,})":
        lambda d: np.asarray(apply_rule(d["monthly_cash_deposits"], CURRENT_THRESHOLD)),
    f"Global £{best_global['threshold']:,.0f}":
        lambda d: np.asarray(apply_rule(d["monthly_cash_deposits"], best_global["threshold"])),
    describe(SPEC_RETAIL, SPEC_CORPORATE, SPEC_VELOCITY):
        lambda d: challenger_flags(d, SPEC_RETAIL, SPEC_CORPORATE, SPEC_VELOCITY),
}

rows = []
for name, rule in candidates.items():
    for quarter in ["Q1", "Q2", "Q3", "Q4"]:
        part = population[population["quarter"] == quarter]
        metrics = classification_metrics(part["case"], rule(part))
        rows.append({"option": name, "quarter": quarter,
                     "alerts_per_month": metrics["alerts"] / 3,
                     "precision": metrics["precision"], "recall": metrics["recall"]})
quarterly = pd.DataFrame(rows)
show(quarterly.pivot(index="option", columns="quarter", values="alerts_per_month").reset_index(),
     "Alerts per month by quarter")
show(quarterly.pivot(index="option", columns="quarter", values="precision").reset_index(),
     "\nPrecision by quarter")

# Coefficient of variation: spread relative to level, so options of very
# different volumes are comparable on the same scale.
stability = quarterly.groupby("option")["alerts_per_month"].agg(
    mean="mean", std="std").assign(volume_cv=lambda d: d["std"] / d["mean"])
stability["q4_vs_q1"] = (
    quarterly[quarterly["quarter"] == "Q4"].set_index("option")["alerts_per_month"]
    / quarterly[quarterly["quarter"] == "Q1"].set_index("option")["alerts_per_month"] - 1)
show(stability.reset_index(), "\nVolume stability across the year")

worst = stability["volume_cv"].idxmax()
print(f"\n  Least stable option: {worst}")
print(f"    volume grows {stability.loc[worst, 'q4_vs_q1']:+.0%} from Q1 to Q4")

incumbent_report = stability_report(population, CURRENT_THRESHOLD,
                                    score_col="monthly_cash_deposits", baseline_periods=3)
save(plot_stability_chart(incumbent_report, metric="alerts",
                          title="TM-021 incumbent alert volume across the year"),
     OUT / "week10_incumbent_stability.png")
print(f"\n  Periods breaching control limits (incumbent): "
      f"{int(incumbent_report['any_breach'].sum())} of {len(incumbent_report)}")
print(f"  PSI on cash deposits, final period: {incumbent_report.iloc[-1]['psi']:.2f} "
      f"({incumbent_report.iloc[-1]['band']})")

print("""
  Every option's volume grows through the year, because the population is
  drifting rather than the rules being unstable in themselves. That matters for
  the recommendation: whatever is chosen must be sized against the LATEST
  quarter, not the twelve-month average, or it goes live already over capacity.
""")

# %% [markdown]
# ## 6. Recommendation

# %%
banner("6a. THE SPEC'S EXAMPLE CALIBRATION, EVALUATED")

example = evaluate_challenger(SPEC_RETAIL, SPEC_CORPORATE, SPEC_VELOCITY)
print(f"""  Retail £{SPEC_RETAIL:,} / Corporate £{SPEC_CORPORATE:,} AND velocity > {SPEC_VELOCITY}

    Alerts     {example['alerts'] / 12:,.0f} per month ({example['alerts'] / 12 / CAPACITY:.0%} of the {CAPACITY:,} allocation)
    Precision  {example['precision']:.2%}   (from {baseline['precision']:.2%})
    Recall     {example['recall']:.2%}   (from {baseline['recall']:.2%})
    Volume     {1 - example['alerts'] / baseline['alerts']:.0%} reduction
""")

print(f"""  It over-corrects, and the capacity column is what shows it.

  Senior management asked for volume reduction and this delivers {1 - example['alerts'] / baseline['alerts']:.0%} of it.
  But it leaves {CAPACITY - example['alerts'] / 12:,.0f} of the {CAPACITY:,} monthly alerts unused -- {1 - example['alerts'] / 12 / CAPACITY:.0%} of the
  investigation capacity the bank is already paying for sits idle -- while
  recall falls {(baseline['recall'] - example['recall']) * 100:.0f} percentage points and {int(example['fn'] - baseline['fn']):,} additional cases go
  unalerted.

  That fails the third constraint. "Missed-risk tolerance limited" is not
  satisfied by a calibration that gives up most of the detection to hit a
  volume target it overshot. Capacity was the binding constraint; a rule using
  {example['alerts'] / 12 / CAPACITY:.0%} of it has stopped being constrained by anything.
""")

log.record("6. Recommendation",
           "SPEC EXAMPLE: " + describe(SPEC_RETAIL, SPEC_CORPORATE, SPEC_VELOCITY),
           example,
           operational_implication=f"{example['alerts'] / 12:,.0f} alerts/month -- only "
                                   f"{example['alerts'] / 12 / CAPACITY:.0%} of the allocation, leaving "
                                   f"{CAPACITY - example['alerts'] / 12:,.0f}/month of paid capacity idle.",
           assumptions="Velocity populated for all customers; segment assignment stable.",
           outcome="rejected",
           rationale=f"Over-corrects. Delivers the volume reduction but gives up "
                     f"{(baseline['recall'] - example['recall']) * 100:.0f}pp of recall and leaves most of the "
                     f"investigation capacity unused. Fails the missed-risk constraint.")

# %% [markdown]
# ### 6b. Searching the design space within capacity
#
# The constraint is a budget, not a target to undershoot. Spend it.

# %%
banner("6b. CALIBRATIONS THAT ACTUALLY USE THE ALLOCATION")

import itertools

rows = []
for retail_cut, corporate_cut, velocity_min in itertools.product(
        [10_000, 12_500, 15_000, 20_000, 25_000], [15_000, 20_000, 25_000, 35_000],
        [None, 0, 1, 2, 3]):
    # Corporate must sit at or above Retail. A lower corporate cut is not
    # defensible to an investigator or a validator, whatever it scores.
    if corporate_cut < retail_cut:
        continue
    metrics = evaluate_challenger(retail_cut, corporate_cut, velocity_min)
    q4_metrics = evaluate_challenger(retail_cut, corporate_cut, velocity_min,
                                     population[population["quarter"] == "Q4"])
    rows.append({
        "retail": retail_cut, "corporate": corporate_cut, "velocity_min": velocity_min,
        "alerts_per_month": metrics["alerts"] / 12,
        "q4_alerts_per_month": q4_metrics["alerts"] / 3,
        "precision": metrics["precision"], "recall": metrics["recall"],
        "fpr": metrics["fpr"], "fn": metrics["fn"],
    })

design_space = pd.DataFrame(rows)

# Size on the Q4 run-rate, not the twelve-month average. The population drifted
# through the year, so an option averaging inside capacity across all four
# quarters can still be over it on the day it goes live. Selecting on the annual
# mean is the mistake section 5 warned about, and it is easy to make here.
within = design_space[design_space["q4_alerts_per_month"] <= CAPACITY].sort_values(
    "recall", ascending=False)
show(within.head(8)[["retail", "corporate", "velocity_min", "alerts_per_month",
                     "q4_alerts_per_month", "precision", "recall", "fpr"]],
     f"Best options whose Q4 run-rate fits the {CAPACITY:,}/month allocation")

over_on_q4 = design_space[(design_space["alerts_per_month"] <= CAPACITY)
                          & (design_space["q4_alerts_per_month"] > CAPACITY)]
print(f"\n  {len(over_on_q4)} of {len(design_space)} designs average inside capacity across the year "
      f"but breach it in Q4.\n  Selecting on the annual mean would have shipped one of those.")

best = within.iloc[0]
BEST_RETAIL = int(best["retail"])
BEST_CORPORATE = int(best["corporate"])
BEST_VELOCITY = None if pd.isna(best["velocity_min"]) else int(best["velocity_min"])
proposed = evaluate_challenger(BEST_RETAIL, BEST_CORPORATE, BEST_VELOCITY)

print(f"""
  Best within capacity: Retail £{BEST_RETAIL:,} / Corporate £{BEST_CORPORATE:,} AND velocity > {BEST_VELOCITY}
    {proposed['alerts'] / 12:,.0f} alerts/month ({proposed['alerts'] / 12 / CAPACITY:.0%} of allocation), recall {proposed['recall']:.2%}

  Against the spec's example: {(proposed['recall'] - example['recall']) * 100:+.1f}pp of recall recovered, for
  {proposed['alerts'] / 12 - example['alerts'] / 12:+,.0f} alerts a month the team already has the capacity to work.
""")

# %%
# Does the segmentation earn its keep, or would one global cut do the same job?
# Constrained the SAME way as the adopted option -- Q4 run-rate inside capacity.
# Comparing against a global threshold that breaches Q4 would not be a
# like-for-like test; it would be scoring a feasible option against an
# infeasible one and calling the difference a benefit of segmentation.
velocity_match = (design_space["velocity_min"].isna() if BEST_VELOCITY is None
                  else design_space["velocity_min"] == BEST_VELOCITY)
global_equivalent = design_space[
    (design_space["retail"] == design_space["corporate"])
    & velocity_match
    & (design_space["q4_alerts_per_month"] <= CAPACITY)
].sort_values("recall", ascending=False)

if not global_equivalent.empty:
    simplest = global_equivalent.iloc[0]
    gap = (best["recall"] - simplest["recall"]) * 100
    print(f"""  A one-threshold check before recommending two:

    {describe(int(simplest['retail']), int(simplest['retail']), BEST_VELOCITY)}
      {simplest['alerts_per_month']:,.0f}/month on the year, {simplest['q4_alerts_per_month']:,.0f}/month in Q4, recall {simplest['recall']:.2%}

    The segmented version buys {gap:+.2f}pp of recall over it.
""")
    if abs(gap) < 1.0:
        print(f"""  That is inside noise. The segmentation is not earning its keep here: it
  adds a second threshold to document, approve, monitor and re-tune, and it
  makes segment assignment an AML control, for under a percentage point of
  detection. Week 5's test applies -- and on this population it fails.

  The recommendation below therefore leads with the simpler rule and offers
  the segmented variant as an option, rather than the other way round.
""")
        BEST_RETAIL = BEST_CORPORATE = int(simplest["retail"])
        proposed = evaluate_challenger(BEST_RETAIL, BEST_CORPORATE, BEST_VELOCITY)

# %% [markdown]
# ### 6c. Recommendation

# %%
banner("6c. RECOMMENDATION")

q4 = population[population["quarter"] == "Q4"]
q4_metrics = evaluate_challenger(BEST_RETAIL, BEST_CORPORATE, BEST_VELOCITY, q4)

if BEST_RETAIL == BEST_CORPORATE:
    calibration = f"    All segments  cash deposits > £{BEST_RETAIL:,} in 30 days"
else:
    calibration = (f"    Retail        cash deposits > £{BEST_RETAIL:,} in 30 days\n"
                   f"    Corporate     cash deposits > £{BEST_CORPORATE:,} in 30 days")
velocity_line = ("" if BEST_VELOCITY is None
                 else f"\n    AND           velocity > {BEST_VELOCITY} outbound transactions")

print(f"""  PROPOSED CALIBRATION

{calibration}{velocity_line}
""")

impact = pd.DataFrame([
    {"Metric": "Alerts (per month)", "Before": f"{baseline['alerts'] / 12:,.0f}",
     "After": f"{proposed['alerts'] / 12:,.0f}",
     "Change": f"{proposed['alerts'] / baseline['alerts'] - 1:+.0%}"},
    {"Metric": "Precision", "Before": f"{baseline['precision']:.2%}",
     "After": f"{proposed['precision']:.2%}",
     "Change": f"{(proposed['precision'] - baseline['precision']) * 100:+.2f}pp"},
    {"Metric": "Recall", "Before": f"{baseline['recall']:.2%}",
     "After": f"{proposed['recall']:.2%}",
     "Change": f"{(proposed['recall'] - baseline['recall']) * 100:+.2f}pp"},
    {"Metric": "FPR", "Before": f"{baseline['fpr']:.2%}",
     "After": f"{proposed['fpr']:.2%}",
     "Change": f"{(proposed['fpr'] - baseline['fpr']) * 100:+.2f}pp"},
])
show(impact, "IMPACT")

print(f"""
  Against the three constraints:

    Investigation capacity fixed   {proposed['alerts'] / 12:,.0f}/month against {CAPACITY:,} allocated
                                   ({proposed['alerts'] / 12 / CAPACITY:.0%} of capacity)
                                   Q4 run-rate: {q4_metrics['alerts'] / 3:,.0f}/month ({q4_metrics['alerts'] / 3 / CAPACITY:.0%})
    Volume reduction expected      {1 - proposed['alerts'] / baseline['alerts']:.0%} reduction delivered
    Missed-risk tolerance limited  {int(proposed['fn']):,} cases not alerted, against {int(baseline['fn']):,}
                                   under the incumbent ({int(proposed['fn'] - baseline['fn']):+,} cases)

  The Q4 run-rate is the number to size against, not the twelve-month average.
  The population drifted through the year, so a calibration sized on the annual
  mean goes live already behind.
""")

log.record("6. Recommendation",
           "ADOPTED: " + describe(BEST_RETAIL, BEST_CORPORATE, BEST_VELOCITY),
           proposed,
           operational_implication=f"{proposed['alerts'] / 12:,.0f} alerts/month "
                                   f"({proposed['alerts'] / 12 / CAPACITY:.0%} of allocation); "
                                   f"Q4 run-rate {q4_metrics['alerts'] / 3:,.0f}/month.",
           assumptions="Velocity populated for all customers; SAR-based labels; "
                       "population continues drifting at the observed rate.",
           outcome="adopted",
           rationale=f"Delivers the required volume reduction while spending the allocated "
                     f"capacity rather than undershooting it. Recovers "
                     f"{(proposed['recall'] - example['recall']) * 100:.0f}pp of recall against the "
                     f"example calibration. Recall still falls against the incumbent; that cost "
                     f"is quantified and put to the risk owner explicitly.")

SPEC_RETAIL_USED, SPEC_CORPORATE_USED, SPEC_VELOCITY_USED = BEST_RETAIL, BEST_CORPORATE, BEST_VELOCITY

# %% [markdown]
# ### Risks

# %%
banner("RISKS")

risks = pd.DataFrame([
    ("Potential blind spots",
     "The velocity AND removes high-value, low-frequency cash deposits -- a single "
     "large structured deposit no longer alerts on this rule at all.",
     "Confirm another rule covers single large cash deposits before implementation. "
     "If none does, add an OR leg at a high cash value with no velocity condition."),
    ("Potential blind spots",
     "Recall falls against the incumbent even on the adopted calibration. The cases "
     "lost are disproportionately low-velocity, so a customer making one large cash "
     "deposit a month is less well covered than before.",
     "Confirm another rule covers single large cash deposits. Report the lost-case "
     "count to the risk owner as an explicit acceptance, not as a footnote."),
    ("Data limitations",
     "A null velocity fails the AND closed, silently un-monitoring that customer. "
     "The backtest cannot detect this because the test field is always populated.",
     "Measure velocity field coverage in production BEFORE implementation. Treat "
     "null as velocity-condition-met (fail open) rather than closed."),
    ("Data limitations",
     "Case labels are SAR-based and therefore absent below the incumbent's line. "
     "Measured recall is optimistic by an unquantified margin.",
     "Below-the-line sample against the proposed rule (see extras/atl_btl_testing.py) "
     "before the change is approved."),
    ("Data limitations",
     "Segment assignment becomes an AML control the moment thresholds differ by segment.",
     "Confirm who can change a customer's segment, and add segment changes to the "
     "monitoring pack."),
    ("Monitoring requirements",
     "Volume grew through the year on every option tested; the population is drifting.",
     "Size against the latest quarter, not the annual average. Control limits from "
     "the most recent stable quarter, re-tune on two consecutive breaches."),
    ("Monitoring requirements",
     "Two thresholds and a velocity condition is three parameters to govern, up from one.",
     "Each gets a named owner, a stated rationale and a re-tuning cycle in the paper."),
], columns=["Risk area", "Risk", "Mitigation"])
show(risks, "Risks and mitigations")

# %% [markdown]
# ## Deliverables

# %%
banner("DELIVERABLES")

sweep_for_paper = threshold_sweep(population, "monthly_cash_deposits", "case",
                                  thresholds=SWEEP_THRESHOLDS)
current_row = sweep_for_paper.iloc[0]
proposed_row = pd.Series({**proposed, "threshold": BEST_RETAIL})

paper = mini_tuning_paper(
    rule_name="TM-021 Cash Deposit Structuring",
    current_rule=f"ALERT IF cash_deposits_30d > {CURRENT_THRESHOLD:,}\n"
                 + f"  proposed: {describe(BEST_RETAIL, BEST_CORPORATE, BEST_VELOCITY)}",
    population=population,
    sweep=sweep_for_paper,
    proposed=proposed_row,
    current=current_row,
    author="Dan Hartwig",
    method_notes=[
        f"Fixed investigation capacity: {CAPACITY:,} alerts/month allocated to this rule.",
        "Objective: maximise recall subject to that constraint, with a required volume reduction.",
        "Stability tested independently across Q1-Q4.",
    ],
    limitations=[
        "A null velocity value fails the AND condition closed, silently removing that "
        "customer from monitoring. Field coverage must be confirmed before implementation.",
        "Segment-specific thresholds make segment assignment an AML control.",
        "Volume grew across all four quarters; the calibration is sized against Q4 and "
        "should be re-assessed within two quarters rather than annually.",
    ],
)
paper_path = save_paper(OUT / "week10_capstone_tuning_paper.md", paper)
log_path = log.save(OUT / "week10_tuning_decision_log.md")

print(f"  Tuning paper        {paper_path.name}")
print(f"  Decision log        {log_path.name}  ({len(log)} options recorded)")
print(f"  Charts              week10_alert_volume.png, week10_incumbent_stability.png")

show(log.to_frame()[["step", "option", "alerts", "precision", "recall", "outcome"]],
     "\nThe decision log -- every option tested, including those rejected")

# %%
answer("Senior management asked for volume reduction. You delivered it. Why is the paper not finished?",
       """
Because volume reduction was the request, not the objective. The objective is
to monitor the risk within what the bank can actually investigate, and those
two only coincide when the alerts removed were unproductive.

This recommendation removes productive alerts as well -- recall falls, and the
paper says by how many cases. That number is the price of the request, and it
belongs in the first paragraph rather than in an appendix, because accepting it
is a risk-appetite decision that senior management has to take knowingly.

A paper that reports the volume reduction and buries the recall cost has
answered the question it was asked and not the one it exists to answer. It will
also be found: the recall column is the first thing independent validation
reads.

The honest close is three sentences -- here is the reduction you asked for,
here is the detection it costs, and here is what additional capacity would buy
instead. Then the decision is theirs, made on the evidence, and the decision
log shows every option that was weighed to get there.
""")

banner("END OF THE PROGRAMME")
print("""
You can now:
  * design labelled and proxy-labelled backtests
  * tune transaction-monitoring thresholds using evidence
  * quantify precision/recall trade-offs
  * build alert-volume and risk-yield curves
  * conduct segment-specific calibration
  * identify instability and model drift
  * compare incumbent versus challenger rules
  * produce validation-ready tuning papers
  * execute a complete transaction-monitoring backtest and calibration exercise

Take the method, not the numbers. Every figure here came from a synthetic
generator whose ground truth you were given. Real tuning is done against a
label set that is sparse, late and partly wrong -- which is exactly why the
method's discipline about stating assumptions, bounding what you cannot
measure, and validating out of time is the part that transfers.

And keep the decision log. It is the cheapest habit in the course and the one
that will still be earning its keep three years from now.
""")
