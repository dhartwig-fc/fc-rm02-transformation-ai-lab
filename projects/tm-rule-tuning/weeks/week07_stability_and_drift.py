# %% [markdown]
# # Week 7 -- Stability and Drift Testing
#
# **Concept:** A rule that works in one period may fail later.
#
# **Topics**
#
# * Population stability
# * Behaviour change
# * Data drift
# * Seasonality
#
# **Exercise**
#
# Create three periods: Year 1, Year 2, Year 3. Simulate distribution changes.
#
# Calculate alert rates, precision and recall.
#
# **Python Technique**
#
# ```python
# from scipy.stats import ks_2samp
# ```
#
# **Success criteria.** Identify unstable thresholds and propose mitigants.

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
from scipy.stats import ks_2samp

from tmtuning import (answer, banner, apply_rule, classification_metrics, generate_population,
                      psi, psi_by_period, show, stability_report, threshold_sweep)
from tmtuning.plots import plot_stability_chart, save

OUT = _ROOT / "outputs"
THRESHOLD = 50_000

# %% [markdown]
# ## 1. Create three periods and simulate distribution changes

# %%
banner("1. THREE YEARS, WITH DRIFT AND SEASONALITY")

# 36 monthly periods, grouped into three years. drift_strength makes transaction
# values inflate and the segment mix tilt as the periods advance.
population = generate_population(n=90_000, seed=707, n_periods=36,
                                 start_period="2023-01", drift_strength=1.3)
population["year"] = "Year " + (population["period_index"] // 12 + 1).astype(str)
population["month_of_year"] = population["period_index"] % 12 + 1

# Seasonality layered on top: a recurring intra-year pattern, peaking in the
# fourth quarter. Unlike drift, it returns to where it started each year -- and
# telling the two apart is the whole diagnostic skill of this week.
seasonal = 1.0 + 0.22 * np.sin((population["month_of_year"] - 3) / 12 * 2 * np.pi)
population["monthly_wire_value"] = (population["monthly_wire_value"] * seasonal).round(2)

show(population.groupby("year").agg(
        rows=("case", "size"), cases=("case", "sum"), prevalence=("case", "mean"),
        median_value=("monthly_wire_value", "median"),
        p90_value=("monthly_wire_value", "quantile")).reset_index(),
     "Three yearly periods")

# %% [markdown]
# ## 2. Alert rates, precision and recall by year

# %%
banner("2. RULE PERFORMANCE BY YEAR")

rows = []
for year, part in population.groupby("year"):
    metrics = classification_metrics(
        part["case"], apply_rule(part["monthly_wire_value"], THRESHOLD))
    rows.append({"year": year, "rows": len(part), **metrics})
yearly = pd.DataFrame(rows)
show(yearly[["year", "rows", "alerts", "alert_rate", "tp", "precision", "recall"]],
     f"Rule: monthly outbound wires > £{THRESHOLD:,}")

first, last = yearly.iloc[0], yearly.iloc[-1]
print(f"""
  Alert rate {first['alert_rate']:.2%} -> {last['alert_rate']:.2%}  ({last['alert_rate'] / first['alert_rate'] - 1:+.0%})
  Precision  {first['precision']:.2%} -> {last['precision']:.2%}
  Recall     {first['recall']:.2%} -> {last['recall']:.2%}

The threshold has not moved. Everything above moved underneath it.
""")

# %% [markdown]
# ## 3. Population stability -- the KS test

# %%
banner("3. KOLMOGOROV-SMIRNOV TEST")

year1 = population.loc[population["year"] == "Year 1", "monthly_wire_value"]
year2 = population.loc[population["year"] == "Year 2", "monthly_wire_value"]
year3 = population.loc[population["year"] == "Year 3", "monthly_wire_value"]

rows = []
for label, sample in [("Year 1 vs Year 2", year2), ("Year 1 vs Year 3", year3)]:
    result = ks_2samp(year1, sample)
    rows.append({
        "comparison": label,
        "ks_statistic": result.statistic,
        "p_value": result.pvalue,
        "psi": psi(year1, sample),
        "n_baseline": len(year1),
        "n_comparison": len(sample),
    })

# A control comparison: split Year 1 into two RANDOM halves. Same distribution by
# construction, so whatever this row shows is the test's noise floor at this
# sample size.
#
# The split has to be random. Taking the first and second half of the year by
# position would compare months 1-6 against months 7-12 -- which differ by the
# seasonal pattern and by within-year drift, so it would measure real change and
# report it as the noise floor, understating every comparison above it.
shuffled = year1.sample(frac=1.0, random_state=707).to_numpy()
half = len(shuffled) // 2
control = ks_2samp(shuffled[:half], shuffled[half:])
rows.append({"comparison": "Year 1 vs itself (random split)", "ks_statistic": control.statistic,
             "p_value": control.pvalue, "psi": psi(shuffled[:half], shuffled[half:]),
             "n_baseline": half, "n_comparison": len(shuffled) - half})

show(pd.DataFrame(rows), "Two-sample KS test against the Year 1 baseline")

print("""
The KS statistic is the largest gap between the two cumulative distributions:
0 means identical, 1 means no overlap. Read THAT, not the p-value.

At these sample sizes the p-value is close to useless. KS power grows with n, so
with tens of thousands of records in each sample it returns a vanishing p-value
for any difference at all, including differences far too small to affect a
threshold. Report a significant p-value on 30,000 records and you have shown
that the samples are not literally identical -- which was never in doubt.

The control row is the discipline that keeps this honest: it is the same
distribution split in two, so its KS statistic is the floor. A comparison worth
acting on has to clear it by a wide margin, not merely beat a p-value.
""")

# %%
answer("KS says the distribution has changed. PSI says 'moderate'. Which do you act on?",
       """
Neither on its own, because they answer different questions.

KS is sensitive to a shift anywhere in the distribution, including in the
dense middle where no threshold sits. PSI is binned, so it weights by where
the mass is and is more forgiving of a small shift in a sparse tail.

For threshold tuning the question is narrower than either: has the
distribution moved AROUND MY THRESHOLD? A rule at £50k does not care about a
change at the 10th percentile, however statistically real it is.

So: use KS and PSI to detect that something moved, then go and look at the
alert rate at the actual threshold -- which is section 2, and the only one of
the three that is denominated in the thing the operation experiences.
""")

# %% [markdown]
# ## 4. Seasonality versus drift
#
# Both make a period look different from the baseline. They need opposite
# responses, and the monitoring pack shows them identically.

# %%
banner("4. SEASONALITY VERSUS DRIFT")

monthly = population.groupby(["year", "month_of_year"]).agg(
    alert_rate=("monthly_wire_value", lambda v: (v > THRESHOLD).mean())).reset_index()
pivot = monthly.pivot(index="month_of_year", columns="year", values="alert_rate")
show(pivot.reset_index(), "Alert rate by calendar month, one column per year")

print("""
Read the table two ways.

  DOWN a column: the recurring intra-year shape. It peaks in the same months
  every year and returns to where it started. That is SEASONALITY.

  ACROSS a row: the same calendar month, year on year, rising steadily. That is
  DRIFT.

Confusing them is expensive in both directions. Re-tuning a threshold in
response to a seasonal peak bakes the peak into the baseline, and the rule then
under-alerts for the rest of the year. Dismissing genuine drift as "just
seasonal" leaves the rule stale until someone notices the queue.

The test is simple and worth applying before any re-tune: compare like calendar
months, never consecutive ones.
""")

# %%
q4 = monthly[monthly["month_of_year"].isin([10, 11, 12])].groupby("year")["alert_rate"].mean()
q2 = monthly[monthly["month_of_year"].isin([4, 5, 6])].groupby("year")["alert_rate"].mean()
seasonal_swing = (q4 / q2 - 1).mean()
yoy = (pivot.iloc[:, -1] / pivot.iloc[:, 0] - 1).mean()
print(f"  Average Q4-versus-Q2 swing within a year : {seasonal_swing:+.0%}  (seasonality)")
print(f"  Average Year 3 versus Year 1, same months: {yoy:+.0%}  (drift)")
print("""
  Both are real and they compound. A Q4 in Year 3 combines the seasonal peak
  with three years of drift, which is when a queue that has been coping for
  months suddenly does not.
""")

# %% [markdown]
# ## 5. Which thresholds are unstable?
#
# Instability is a property of the threshold, not only of the population. Two
# thresholds on the same drifting data can behave completely differently.

# %%
banner("5. IDENTIFYING UNSTABLE THRESHOLDS")

rows = []
for candidate in [10_000, 25_000, 50_000, 100_000, 200_000]:
    by_year = []
    for year, part in population.groupby("year"):
        flags = apply_rule(part["monthly_wire_value"], candidate)
        by_year.append({"alert_rate": flags.mean(),
                        "precision": classification_metrics(part["case"], flags)["precision"]})
    rates = [y["alert_rate"] for y in by_year]
    precisions = [y["precision"] for y in by_year]
    rows.append({
        "threshold": candidate,
        "alert_rate_y1": rates[0], "alert_rate_y3": rates[-1],
        "alert_rate_growth": rates[-1] / rates[0] - 1,
        "precision_y1": precisions[0], "precision_y3": precisions[-1],
        # Coefficient of variation: spread relative to level, so thresholds of
        # very different volumes can be compared on the same scale.
        "volume_cv": np.std(rates) / np.mean(rates),
    })
instability = pd.DataFrame(rows).sort_values("volume_cv", ascending=False)
show(instability, "Stability of each candidate threshold across the three years")

worst = instability.iloc[0]
print(f"""
The £{worst['threshold']:,.0f} threshold is the least stable: its alert volume grows
{worst['alert_rate_growth']:+.0%} across the three years.

Why thresholds differ. A threshold sitting where the distribution is STEEP --
in the dense body -- converts a small shift in customer behaviour into a large
change in alert volume. One further out sits where the curve is flat, so the
same shift barely moves it. Two thresholds with identical precision today can
carry entirely different volume risk, and nothing in a single-period backtest
shows it.
""")

# %% [markdown]
# ## 6. Control charts and PSI over the full 36 periods

# %%
banner("6. CONTROL CHARTS")

report = stability_report(population, THRESHOLD, baseline_periods=12)
save(plot_stability_chart(report, metric="alerts",
                          title="Alert volume across three years"),
     OUT / "week07_volume_control_chart.png")
save(plot_stability_chart(report, metric="precision", limits_key="precision_limits",
                          title="Precision across three years"),
     OUT / "week07_precision_control_chart.png")

limits = report.attrs["volume_limits"]
print(f"  Limits from the 12 baseline periods (Year 1): "
      f"centre {limits['centre']:,.0f}, band {limits['lower']:,.0f}-{limits['upper']:,.0f}")
print(f"  Periods breaching a control limit: {int(report['any_breach'].sum())} of {len(report)}")
print(f"  Charts written to {OUT}")

print("""
Limits are anchored to Year 1 -- the window the threshold was tuned on. Deriving
them from all 36 periods would let the drift inflate its own mean and standard
deviation until the limits chase the trend and nothing ever breaches. A control
chart that cannot fire is assurance theatre.
""")

# %%
show(psi_by_period(population, "monthly_wire_value", "period").iloc[::4],
     "PSI against the first period (every 4th period)")

# %% [markdown]
# ## 7. Diagnosing what moved

# %%
banner("7. DIAGNOSIS")

diagnosis = pd.DataFrame([
    ("Population drift", "Rises steadily", "Moves, same direction", "Roughly held",
     "Customers changed. Re-tune to the new distribution."),
    ("Seasonality", "Rises and returns", "Cyclical, repeating", "Roughly held",
     "Do not re-tune. Adjust capacity planning, or de-seasonalise the input."),
    ("Rule decay", "Flat", "Flat", "Falls",
     "Same customers, worse yield. Typology moved or the rule is being evaded."),
    ("Label drift", "Flat", "Flat", "Falls",
     "Check investigation standards and SAR policy before blaming the rule."),
    ("Upstream data break", "Jumps at one period", "Jumps or collapses", "Erratic",
     "Not a tuning problem. Check the feed, not the threshold."),
], columns=["Pattern", "PSI / KS", "Alert volume", "Precision", "What it means and what to do"])
show(diagnosis, "Read the signals together, never one alone")

print(f"""  Our case: KS {rows[0] if False else ''}""".rstrip())
print(f"  Our case: KS statistic {ks_2samp(year1, year3).statistic:.3f} against Year 1, "
      f"alert rate {first['alert_rate']:.2%} -> {last['alert_rate']:.2%}, "
      f"precision {first['precision']:.2%} -> {last['precision']:.2%}.")
print("""  Distribution moved, volume up sharply, precision broadly held, and the
  movement does not return within the year -> POPULATION DRIFT, with a
  seasonal component layered on top.

  The rule logic is sound. It is being pointed at a population that has moved
  past its threshold. Re-tune the number; do not redesign the rule.
""")

# %% [markdown]
# ## 8. Success criteria: propose mitigants

# %%
banner("8. MITIGANTS FOR UNSTABLE THRESHOLDS")

mitigants = pd.DataFrame([
    ("Re-tune on a schedule", "Fixed threshold, reviewed quarterly rather than annually",
     "Simple; fits existing governance", "Always lags the drift by up to a quarter"),
    ("Shorten the tuning window", "Tune on the most recent 12 months, not all history",
     "Threshold reflects current behaviour", "Noisier; less data behind the estimate"),
    ("Percentile-based threshold", "Alert on the top N% rather than a fixed £ value",
     "Alert volume is stable by construction", "Risk appetite drifts silently with the "
     "population; needs its own monitoring"),
    ("Index the threshold", "Uprate by an external index (inflation, portfolio average)",
     "Transparent, defensible, auditable", "The index may not track the actual drift"),
    ("De-seasonalise the input", "Compare against the same month last year",
     "Separates seasonality from drift cleanly", "Needs 24+ months of history"),
    ("Segment the rule", "Separate thresholds per segment (Week 5)",
     "Isolates drift to the segment causing it", "More parameters to govern and monitor"),
    ("Tighten monitoring triggers", "Control limits with a named action and owner",
     "Detects the next drift earlier", "Detection only -- does not fix the threshold"),
], columns=["Mitigant", "What it is", "Strength", "Weakness"])
show(mitigants, "Options, with what each costs")

print("""
The percentile-based option deserves a warning, because it is the one that
looks most attractive and hides the most.

Alerting on "the top 1% by value" holds alert volume constant forever, which
solves the operational problem completely. It also means the effective pound
threshold moves every period with no decision, no paper and no approval. Risk
appetite becomes an emergent property of the population rather than something
the bank set. If you adopt it, the monitoring has to track the IMPLIED pound
threshold and trigger a review when it moves beyond an agreed band -- otherwise
you have not removed the drift, only stopped measuring it.
""")

print(f"""  RECOMMENDED FOR THIS RULE

  1. Shorten the tuning window to the most recent 12 periods. The population has
     moved {yoy:+.0%} year on year at like-for-like months, so a threshold fitted to
     all three years is stale on the day it goes live.

  2. Compare like calendar months when assessing the next move, so the {seasonal_swing:+.0%}
     Q4 swing is not mistaken for further drift.

  3. Set control limits from the most recent stable 12 periods, with a named
     action: re-tune when alert volume breaches for two consecutive periods.

  4. Avoid the £{worst['threshold']:,.0f} region. It sits on the steep part of the
     distribution, so it converts small behavioural shifts into large volume
     swings -- the flatter neighbouring thresholds cost little detection and are
     materially more stable.
""")

banner("END OF WEEK 7")
print("""
Carry forward into Week 8:
  * Read the KS statistic, not the p-value -- at scale the p-value always fires.
  * Always run a same-distribution control to establish the noise floor.
  * Compare like calendar months, or seasonality reads as drift.
  * Instability is a property of the threshold's position on the curve.
  * Every mitigant has a cost. Percentile thresholds hide the drift rather than
    removing it.
""")
