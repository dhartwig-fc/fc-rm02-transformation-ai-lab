# %% [markdown]
# # Week 10 -- Capstone: Complete Backtest and Calibration Exercise
#
# **The brief.**
#
# > Rule TM-014 (*monthly outbound wires > £50,000*) has been in production for
# > two years without re-tuning. Operations report the alert queue is growing
# > and investigators say yield has fallen. Capacity is 2,500 alerts per month
# > and there is no budget for more.
# >
# > Determine whether the rule is still fit for purpose, recommend a
# > calibration, and produce a paper for independent validation.
#
# This runs the whole method end to end. Nothing new is introduced -- every
# step is a week you have already done.
#
# **Success criteria.** A defensible recommendation, evidence for each of the
# ten validation areas from Week 9, and a generated paper.

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

from tmtuning import (answer, banner, apply_rule, btl_test, calibrate_segments,
                      classification_metrics, compare_rules, compare_uniform_vs_segmented,
                      generate_population, marginal_yield, optimise_threshold, required_sample_size,
                      rule_overlap, save_paper, segment_summary, show, split_by_period,
                      stability_report, threshold_sweep, tuning_paper)
from tmtuning.plots import (plot_alert_volume_curve, plot_precision_recall_tradeoff,
                            plot_risk_yield_curve, plot_segment_curves, plot_stability_chart, save)
from tmtuning.segments import segment_sweeps

OUT = _ROOT / "outputs"
# Capacity allocated to THIS rule, not the whole team. The Week 4 team of 12
# investigators can work 12 * 25 * 21 = 6,300 alerts a month across all 18
# rules in the estate; TM-014's agreed share of that queue is 350. Tuning a
# single rule against the whole team's capacity is a common and expensive
# error -- it implicitly assumes every other rule stops firing.
INVESTIGATORS = 12
ALERTS_PER_DAY_EACH = 25
WORKING_DAYS = 21
RULES_IN_ESTATE = 18
TEAM_CAPACITY = INVESTIGATORS * ALERTS_PER_DAY_EACH * WORKING_DAYS
CAPACITY = 350          # TM-014's allocated share of the monthly queue
CURRENT_THRESHOLD = 10_000   # cash deposits > £10k in 30 days
SCORE_COL = "monthly_cash_deposits"

# %% [markdown]
# ## Step 1 -- Build the sample and profile it

# %%
banner("STEP 1: SAMPLE AND PROFILE")

print(f"  Team capacity     {TEAM_CAPACITY:,} alerts/month "
      f"({INVESTIGATORS} investigators x {ALERTS_PER_DAY_EACH}/day x {WORKING_DAYS} days)")
print(f"  Rules in estate   {RULES_IN_ESTATE}")
print(f"  TM-021 allocation {CAPACITY:,} alerts/month "
      f"({CAPACITY / TEAM_CAPACITY:.1%} of the team's queue)\n")

# Two years of data, with drift -- which is the complaint being investigated.
population = generate_population(n=60_000, seed=1010, n_periods=24,
                                 start_period="2024-01", drift_strength=1.4)
in_time, out_of_time = split_by_period(population, holdout_periods=6)

print(f"  Sample          {len(population):,} customer-months over 24 periods")
print(f"  True cases      {population['case'].sum():,} ({population['case'].mean():.2%} base rate)")
print(f"  Tuning window   {in_time['period'].min()} to {in_time['period'].max()} ({len(in_time):,} rows)")
print(f"  Holdout         {out_of_time['period'].min()} to {out_of_time['period'].max()} ({len(out_of_time):,} rows)")

show(segment_summary(population, value_col=SCORE_COL).reset_index(), "\nSegment profile")

# %% [markdown]
# ## Step 2 -- Backtest the incumbent

# %%
banner("STEP 2: INCUMBENT PERFORMANCE")

current_flags = apply_rule(in_time[SCORE_COL], CURRENT_THRESHOLD)
current_metrics = classification_metrics(in_time["case"], current_flags)

print(f"  Rule TM-021 at £{CURRENT_THRESHOLD:,} over the tuning window:\n")
monthly_now = current_metrics["alerts"] / in_time["period"].nunique()
print(f"    Alerts              {current_metrics['alerts']:,} over {in_time['period'].nunique()} periods")
print(f"    Monthly average     {monthly_now:,.0f}  (capacity {CAPACITY:,})")
print(f"    Capacity position   {monthly_now / CAPACITY - 1:+.0%} against capacity")
print(f"    Precision           {current_metrics['precision']:.2%}")
print(f"    Recall              {current_metrics['recall']:.2%}")
print(f"    Effort per case     {current_metrics['alerts_per_true_positive']:.1f} alerts")

incumbent_stability = stability_report(population, CURRENT_THRESHOLD, score_col=SCORE_COL, baseline_periods=6)
save(plot_stability_chart(incumbent_stability, metric="alerts",
                          title="TM-014 alert volume -- incumbent threshold"),
     OUT / "week10_incumbent_stability.png")

first_six = incumbent_stability.head(6)["alerts"].mean()
last_six = incumbent_stability.tail(6)["alerts"].mean()
print(f"\n  Monthly alert volume: {first_six:,.0f} (first 6 periods) -> "
      f"{last_six:,.0f} (last 6)  [{last_six / first_six - 1:+.0%}]")
print(f"  Periods breaching control limits: "
      f"{int(incumbent_stability['any_breach'].sum())} of {len(incumbent_stability)}")
print(f"  PSI in final period: {incumbent_stability.iloc[-1]['psi']:.2f} "
      f"({incumbent_stability.iloc[-1]['band']})")

print("""
  FINDING: Alert volume has grown well past capacity while precision has
  broadly held. PSI is in the significant band. That signature -- population
  moved, rule logic still sound -- is population drift, not rule decay. The
  threshold is stale rather than wrong in kind, so re-tuning is the right
  remedy (Week 7).
""")

# %% [markdown]
# ## Step 3 -- Sweep and select under the capacity constraint

# %%
banner("STEP 3: THRESHOLD SELECTION")

sweep = threshold_sweep(in_time, SCORE_COL, "case", n_thresholds=50)
periods_in_time = in_time["period"].nunique()
budget = CAPACITY * periods_in_time  # capacity is monthly; the sweep spans the window

recommended = optimise_threshold(sweep, objective="recall", max_alerts=budget)
print(f"  Constraint: <= {CAPACITY:,} alerts/month x {periods_in_time} periods = {budget:,}")
print(f"  Objective : maximise recall\n")
print(f"    Recommended threshold  £{recommended['threshold']:,.0f}")
print(f"    Alerts                 {int(recommended['alerts']):,} "
      f"({recommended['alerts'] / periods_in_time:,.0f} per month)")
print(f"    Precision              {recommended['precision']:.2%}")
print(f"    Recall                 {recommended['recall']:.2%}")

marginal = marginal_yield(sweep)
at_point = marginal.loc[(marginal["threshold"] - recommended["threshold"]).abs().idxmin()]
print(f"    Marginal precision     {at_point['marginal_precision']:.2%} "
      f"(base rate {in_time['case'].mean():.2%})")

for name, fig in [
    ("week10_alert_volume.png", plot_alert_volume_curve(sweep, capacity=budget)),
    ("week10_risk_yield.png", plot_risk_yield_curve(sweep)),
    ("week10_precision_recall.png", plot_precision_recall_tradeoff(sweep)),
]:
    save(fig, OUT / name)

# %% [markdown]
# ## Step 4 -- Out-of-time validation

# %%
banner("STEP 4: OUT-OF-TIME VALIDATION")

threshold = float(recommended["threshold"])
oot = pd.DataFrame([
    {"sample": "In-time",
     **classification_metrics(in_time["case"], apply_rule(in_time[SCORE_COL], threshold))},
    {"sample": "Out-of-time",
     **classification_metrics(out_of_time["case"], apply_rule(out_of_time[SCORE_COL], threshold))},
])
show(oot[["sample", "alerts", "tp", "precision", "recall", "alert_rate", "alerts_per_true_positive"]],
     f"Threshold £{threshold:,.0f} on both samples (compare RATES, not counts)")

change = (oot.loc[1, "precision"] - oot.loc[0, "precision"]) * 100
print(f"\n  Precision change out of time: {change:+.2f}pp")

oot_rate = oot.loc[1, "alert_rate"] * len(out_of_time) / out_of_time["period"].nunique()
print(f"  Implied monthly volume in the holdout: {oot_rate:,.0f} (capacity {CAPACITY:,})")
if oot_rate > CAPACITY:
    print("\n  WARNING: the threshold fits capacity in the tuning window but NOT in the")
    print("  holdout, because drift continued. Tune on the most recent periods, not")
    print("  the full history, when the population is known to be moving.")

# %% [markdown]
# ## Step 5 -- Segment calibration

# %%
banner("STEP 5: SEGMENT CALIBRATION")

segment_comparison = compare_uniform_vs_segmented(in_time, capacity=budget, score_col=SCORE_COL)
show(segment_comparison[["segment", "threshold_uniform", "alerts_uniform", "tp_uniform",
                         "threshold_segmented", "alerts_segmented", "tp_segmented", "uplift_tp"]],
     f"Same {budget:,} alert budget, allocated two ways")

total = segment_comparison[segment_comparison["segment"] == "TOTAL"].iloc[0]
uplift_pct = total["uplift_tp"] / max(total["tp_uniform"], 1)
print(f"\n  Detection uplift from segmentation: {int(total['uplift_tp']):+,} cases "
      f"({uplift_pct:+.1%}) at the same alert volume")

save(plot_segment_curves(segment_sweeps(in_time, score_col=SCORE_COL)), OUT / "week10_segment_curves.png")

# %% [markdown]
# ## Step 6 -- Below-the-line testing

# %%
banner("STEP 6: BELOW-THE-LINE TESTING")

planned_n = required_sample_size(expected_rate=0.02, margin_of_error=0.01, confidence=0.95)
print(f"  Sample size planned for +/-1pp at 95% on an expected 2% rate: {planned_n:,} records")

recommended_flags = apply_rule(in_time[SCORE_COL], threshold)
btl = btl_test(in_time, recommended_flags, n_below=planned_n, seed=1010)

print(f"\n  Below-the-line population   {btl['below_the_line_population']:,}")
print(f"  Reviewed                    {btl['sampled']:,}")
print(f"  True cases found            {btl['productive_in_sample']:,}")
print(f"  Rate                        {btl['observed_rate']:.2%} "
      f"(95% CI {btl['rate_lower']:.2%} to {btl['rate_upper']:.2%})")
print(f"  Estimated missed cases      {btl['estimated_missed_cases']:,.0f}")
print(f"  Upper bound                 {btl['estimated_missed_cases_upper']:,.0f}  <- quote this one")

# %% [markdown]
# ### Incremental impact: measure it, do not sample it
#
# The sampled estimate above bounds the *absolute* missed risk below the new
# line. It is the wrong tool for the *incremental* question -- what this change
# costs relative to today -- and using it there produces nonsense.

# %%
banner("STEP 6b: INCREMENTAL RISK OF THE CHANGE")

naive_current = btl_test(in_time, current_flags, n_below=planned_n, seed=1010)
naive_delta = btl["estimated_missed_cases_upper"] - naive_current["estimated_missed_cases_upper"]
# Width of each sampled estimate, to compare against the effect being measured.
naive_width = (btl["rate_upper"] - btl["rate_lower"]) * btl["below_the_line_population"]
tightening = threshold > CURRENT_THRESHOLD

print("  Naive approach -- two separate BTL samples:")
print(f"    incumbent upper bound   {naive_current['estimated_missed_cases_upper']:,.0f}")
print(f"    recommended upper bound {btl['estimated_missed_cases_upper']:,.0f}")
print(f"    apparent difference     {naive_delta:+,.0f}")
print(f"    width of ONE estimate's own 95% interval: {naive_width:,.0f} cases")

if tightening and naive_delta < 0:
    print("""
  That difference has the wrong SIGN. The recommendation tightens the
  threshold, so it must miss at least as much as the incumbent -- a strict
  superset of records falls below the line. A negative number here is simply
  impossible.""")
elif tightening:
    print("""
  The sign happens to be right this time. Do not take any comfort from that:
  the recommendation tightens the threshold, so a strict superset of records
  falls below the line and the difference COULD NOT have been negative in
  truth. A correct sign here is the sampling error landing the right way, not
  evidence the estimate is sound.""")
else:
    print("""
  The recommendation loosens the threshold, so fewer records fall below the
  line and the difference should be negative.""")

print(f"""
  Either way the estimate is unusable, and this is the line that shows why:
  the difference being measured is {abs(naive_delta):,.0f} cases, while the 95% interval
  around ONE of the two estimates is {naive_width:,.0f} cases wide. The noise is
  {naive_width / max(abs(naive_delta), 1):.1f}x the signal.

  Never difference two sampled estimates when the effect is smaller than
  either one's confidence interval.
""")

# %%
deposits = in_time[SCORE_COL]
band = in_time[(deposits > CURRENT_THRESHOLD) & (deposits <= threshold)]
band_cases = int(band["case"].sum())

print(f"  Correct approach -- measure the band directly:\n")
print(f"    Records between £{CURRENT_THRESHOLD:,} and £{threshold:,.0f}: {len(band):,}")
print(f"    True cases among them:                         {band_cases:,}")
print(f"    Precision of the alerts being given up:        "
      f"{band_cases / len(band) if len(band) else float('nan'):.2%}")
print(f"""
  No sampling, no confidence interval, no estimate. These records alerted
  under the incumbent rule, so they were investigated and they carry real
  dispositions. The cost of tightening a threshold is directly observable
  from historical alert outcomes.

  The distinction generalises, and it is worth holding on to:

    TIGHTENING a threshold -- the records you stop alerting on are above the
    historical line. They have labels. COUNT them.

    LOOSENING a threshold -- the records you start alerting on were never
    investigated. They have no labels. SAMPLE them, and report an interval.

  Sampling where you could have counted spends review effort to obtain a
  worse answer, and invites exactly the sign error shown above.
""")

delta = band_cases
print(f"  Incremental missed cases from this change: +{delta:,} over "
      f"{periods_in_time} periods ({delta / periods_in_time:.1f} per month)")
print(f"  Alert volume saved: {int(current_metrics['alerts'] - recommended['alerts']):,} "
      f"({(current_metrics['alerts'] - recommended['alerts']) / periods_in_time:,.0f} per month)")
print(f"  Price of the change: {(current_metrics['alerts'] - recommended['alerts']) / max(delta, 1):,.0f} "
      f"alerts saved per case given up")
print("\n  That ratio is the whole recommendation in one number, and it is the")
print("  sentence the risk owner has to sign.")

# %% [markdown]
# ## Step 7 -- Challenger

# %%
banner("STEP 7: CHALLENGER COMPARISON")

challenger_flags = (
    apply_rule(in_time[SCORE_COL], threshold)
    | ((in_time[SCORE_COL] > threshold * 0.5) & (in_time["high_risk_jurisdiction"] == 1))
    | ((in_time[SCORE_COL] > threshold * 0.5) & (in_time["pep_flag"] == 1))
)
rules = {
    "TM-021 incumbent": current_flags,
    "TM-021 re-tuned": recommended_flags,
    "TM-021 + risk overlay": np.asarray(challenger_flags),
}
comparison = compare_rules(in_time, rules, champion="TM-021 incumbent")
show(comparison[["alerts", "tp", "precision", "recall", "alerts_per_true_positive",
                 "delta_tp", "delta_alerts"]].reset_index(), "Three options on the same sample")

overlap = rule_overlap(in_time, recommended_flags, np.asarray(challenger_flags))
show(overlap, "\nRe-tuned versus risk overlay -- where they disagree")

# %% [markdown]
# ## Step 8 -- Recommendation and paper

# %%
banner("STEP 8: RECOMMENDATION")

risk_sentence = (
    f"Accepts {delta:,} additional missed cases ({delta / periods_in_time:.1f}/month), "
    f"measured directly from historical dispositions."
)

print(f"""
  RECOMMENDATION

  1. Re-tune TM-021 from £{CURRENT_THRESHOLD:,} to £{threshold:,.0f}.
     Monthly volume {monthly_now:,.0f} -> {recommended['alerts'] / periods_in_time:,.0f}, within the {CAPACITY:,}/month capacity.
     {risk_sentence}

  2. Adopt segment-specific thresholds.
     {int(total['uplift_tp']):+,} cases ({uplift_pct:+.1%}) at the same alert budget.
     Conditional on segment data quality being confirmed as a monitoring control.

  3. Re-tune against the most recent 12 periods, not the full 24.
     The population is drifting; a threshold fitted to two years of history
     is already stale on the day it is implemented.

  4. Implement the Week 7 monitoring triggers, and re-tune when alert volume
     breaches its control band for two consecutive periods rather than
     waiting for the annual cycle.

  NOT recommended at this stage: the risk overlay challenger. It improves
  detection, but it depends on jurisdiction and PEP flags whose data quality
  has not been assessed. A condition on a field that is null for part of the
  population fails closed and silently under-monitors exactly the customers
  it was added to catch (Week 6).
""")

# %%
# Score the live threshold exactly rather than snapping it to the grid.
current_row = threshold_sweep(in_time, SCORE_COL, "case",
                             thresholds=[CURRENT_THRESHOLD]).iloc[0]

paper = tuning_paper(
    rule_name="TM-021 Cash Deposit Structuring",
    rule_logic=f"ALERT IF cash_deposits_30d > {CURRENT_THRESHOLD:,}\n"
               f"  proposed: cash_deposits_30d > {threshold:,.0f}\n"
               f"  scope: all active customers | frequency: monthly, rolling 30 days",
    population=in_time,
    sweep=sweep,
    proposed=recommended,
    current=current_row,
    btl=btl,
    segment_comparison=segment_comparison,
    stability=incumbent_stability,
    challenger=comparison,
    overlap=overlap,
    author="Dan Hartwig",
    label_basis="Confirmed SAR/STR submission within 90 days of the alert period",
    limitations=[
        f"The population is drifting (PSI {incumbent_stability.iloc[-1]['psi']:.2f} in the "
        "final period). The recommended threshold fits capacity in the tuning window but "
        "is projected to exceed it within the holdout, so it should be treated as valid "
        "for two quarters and re-assessed, not set annually.",
        "Segment data quality has not been assessed. Segment-specific thresholds make "
        "segment assignment an AML control, and that dependency must be confirmed before "
        "recommendation 2 is implemented.",
        "The risk-overlay challenger was not recommended because field coverage for "
        "jurisdiction and PEP flags is unverified, not because it underperformed.",
    ],
)
path = save_paper(OUT / "week10_capstone_tuning_paper.md", paper)

banner("ARTEFACTS PRODUCED")
for artefact in sorted(OUT.glob("week10_*")):
    print(f"  {artefact.name}")

# %%
answer("The rule now fires fewer alerts and finds fewer cases. How is that an improvement?",
       """
Because the alternative was not "the same rule, working". It was a queue
above capacity, and alerts above capacity are not investigated -- they age.

An alert that is never worked has the same detection value as an alert never
raised, but it carries extra cost: it consumes triage, it inflates the
apparent coverage of the rule, and it lets the bank report monitoring it is
not performing. The recommendation converts a rule that nominally detects
more risk into one that actually detects what it raises.

That argument only holds while capacity really is fixed. If the true
recommendation is "fund four more investigators", the capacity frontier from
Week 4 is how you make that case -- and it should be put alongside this one,
not instead of it. A tuning paper that silently accepts a resourcing
constraint it was never asked to accept has made a risk decision on the
bank's behalf.
""")

banner("END OF THE PROGRAMME")
print("""
You can now:
  * design labelled and proxy-labelled backtests
  * tune thresholds using evidence rather than judgement alone
  * quantify precision/recall trade-offs
  * build alert-volume and risk-yield curves
  * conduct segment-specific calibration
  * identify instability and model drift
  * compare incumbent versus challenger rules
  * produce validation-ready tuning papers
  * execute a complete backtest and calibration exercise

Take the method, not the numbers. Every figure here came from a synthetic
generator whose ground truth you were given. Real tuning is done against a
label set that is sparse, late and partly wrong -- which is exactly why the
method's discipline about stating assumptions, bounding what you cannot
measure, and validating out of time is the part that transfers.
""")
