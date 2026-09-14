# %% [markdown]
# # Week 7 -- Stability, Instability and Model Drift
#
# **Concept:** A threshold is tuned against one snapshot of behaviour.
# Behaviour then moves. Detect that before the alert backlog or the regulator
# does.
#
# **Topics**
#
# * Period-by-period performance versus a pooled figure
# * Population Stability Index (PSI)
# * Control charts, and where the limits must come from
# * Distinguishing population drift from rule decay
# * Setting monitoring triggers that can actually fire
#
# **Success criteria.** Detect drift in a population you were not told was
# drifting, and say which kind it is.

# %%
# --- path bootstrap ---
import pathlib
import sys

for _p in pathlib.Path(__file__ if "__file__" in globals() else "x").resolve().parents:
    if (_p / "src" / "tmtuning").is_dir():
        sys.path.insert(0, str(_p / "src"))
        _ROOT = _p
        break

import pandas as pd

from tmtuning import (answer, banner, generate_population, psi, psi_by_period, show,
                      stability_report)
from tmtuning.plots import plot_stability_chart, save

OUT = _ROOT / "outputs"
THRESHOLD = 50_000

# %% [markdown]
# ## 1. A pooled figure hides everything interesting

# %%
banner("1. POOLED VERSUS PERIODIC")

stable_pop = generate_population(n=36_000, seed=707, n_periods=12, drift_strength=0.0)
drifting_pop = generate_population(n=36_000, seed=707, n_periods=12, drift_strength=1.6)

for name, pop in [("Stationary", stable_pop), ("Drifting", drifting_pop)]:
    report = stability_report(pop, THRESHOLD, baseline_periods=3)
    pooled_precision = report["tp"].sum() / report["alerts"].sum()
    print(f"{name:<12} pooled precision {pooled_precision:.2%}   "
          f"pooled alerts {int(report['alerts'].sum()):,}")

print("""
Two populations, similar pooled precision. One is stable and one has doubled
its alert volume over the year. The pooled number cannot tell them apart,
which is why a backtest that reports a single figure has not finished the job.
""")

# %% [markdown]
# ## 2. Period-by-period

# %%
banner("2. PERIOD-BY-PERIOD PERFORMANCE")

stable_report = stability_report(stable_pop, THRESHOLD, baseline_periods=3)
drift_report = stability_report(drifting_pop, THRESHOLD, baseline_periods=3)

show(stable_report[["period", "is_baseline", "alerts", "tp", "precision", "recall", "psi", "band"]],
     "Stationary population")
show(drift_report[["period", "is_baseline", "alerts", "tp", "precision", "recall", "psi", "band"]],
     "\nDrifting population")

first, last = drift_report.iloc[0], drift_report.iloc[-1]
print(f"\nDrifting: alert volume {int(first['alerts']):,} -> {int(last['alerts']):,} "
      f"({last['alerts'] / first['alerts'] - 1:+.0%}) over 12 periods.")

# %% [markdown]
# ## 3. Population Stability Index

# %%
banner("3. PSI")

print("""PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Bin edges come from the BASELINE distribution's quantiles, so the baseline is
uniform across bins by construction and any departure belongs to the
comparison period.

Conventional bands:  < 0.10 stable   |   0.10-0.25 moderate   |   > 0.25 significant
""")

psi_compare = pd.merge(
    psi_by_period(stable_pop, "monthly_wire_value").rename(columns={"psi": "psi_stable", "band": "band_stable"}),
    psi_by_period(drifting_pop, "monthly_wire_value").rename(columns={"psi": "psi_drift", "band": "band_drift"}),
    on="period", suffixes=("_s", "_d"),
)
show(psi_compare[["period", "psi_stable", "band_stable", "psi_drift", "band_drift"]],
     "PSI against the first period")

print("\nPSI measures the INPUT distribution only. It knows nothing about")
print("outcomes -- a population can shift with no effect on detection, and")
print("detection can decay with no population shift. Never use PSI alone.")

# %% [markdown]
# ## 4. Control charts -- and where the limits come from

# %%
banner("4. CONTROL CHARTS")

save(plot_stability_chart(drift_report, metric="alerts",
                          title="Alert volume stability -- drifting population"),
     OUT / "week07_volume_control_chart.png")
save(plot_stability_chart(drift_report, metric="precision", limits_key="precision_limits",
                          title="Precision stability -- drifting population"),
     OUT / "week07_precision_control_chart.png")
print(f"Charts written to {OUT}")

limits = drift_report.attrs["volume_limits"]
print(f"\nVolume limits from the {int(drift_report['is_baseline'].sum())} baseline periods: "
      f"centre {limits['centre']:,.0f}, "
      f"band {limits['lower']:,.0f} to {limits['upper']:,.0f} "
      f"({limits['k']:g} sigma, {limits['method']})")

print(f"""
Two design choices worth defending in writing:

1. Limits are anchored to the BASELINE window -- the periods the threshold was
   tuned on. Deriving them from the whole series lets a steadily drifting
   metric inflate its own mean and standard deviation until the limits chase
   the drift and nothing ever breaches. A control chart that cannot fire is
   assurance theatre.

2. Alert VOLUME uses Poisson limits (centre +/- k*sqrt(centre)), because the
   variance of a count is tied to its level. Precision, a proportion, uses
   normal limits. Using normal limits on counts gives limits that are too wide
   at low volumes and too narrow at high ones.

Breaches -- stationary: {int(stable_report['any_breach'].sum())} of {len(stable_report)} periods.
Breaches -- drifting:   {int(drift_report['any_breach'].sum())} of {len(drift_report)} periods.
""")

# %% [markdown]
# ## 5. Population drift versus rule decay

# %%
banner("5. DIAGNOSING WHAT MOVED")

diagnosis = pd.DataFrame([
    ("Population drift", "PSI rises", "Volume moves", "Precision roughly held",
     "Customers changed. Re-tune the threshold to the new distribution."),
    ("Rule decay", "PSI flat", "Volume flat", "Precision falls",
     "Same customers, worse yield. Typology has moved or the rule is being evaded."),
    ("Label drift", "PSI flat", "Volume flat", "Precision falls",
     "Check investigation standards and SAR policy before blaming the rule."),
    ("Upstream data break", "PSI jumps at one period", "Volume jumps or collapses", "Erratic",
     "Not a tuning problem. Check the feed, not the threshold."),
], columns=["Pattern", "PSI", "Alert volume", "Precision", "What it means and what to do"])
show(diagnosis, "Read the three signals together")

first_prec, last_prec = drift_report.iloc[0]["precision"], drift_report.iloc[-1]["precision"]
print(f"\nOur drifting case: PSI {drift_report.iloc[-1]['psi']:.2f} (significant), "
      f"volume {int(drift_report.iloc[0]['alerts']):,} -> {int(drift_report.iloc[-1]['alerts']):,}, "
      f"precision {first_prec:.2%} -> {last_prec:.2%}.")
print("PSI up, volume up, precision broadly held -> POPULATION DRIFT.")
print("The rule still works; it is being pointed at a population that has moved")
print("past its threshold. The fix is re-tuning, not redesigning.")

answer("Why does that distinction change what you do next?",
       """
Because the two failures have opposite remedies and similar symptoms in a
monitoring pack.

Population drift means the threshold is stale: customers now transact at
levels that were exceptional when it was set, so volume climbs while the
rule's logic remains sound. Re-tune the number.

Rule decay means the threshold is fine but the logic no longer describes the
risk -- a typology has moved, or the rule's behaviour has been learned and is
being structured around. Re-tuning the number here achieves nothing except
moving volume; the rule needs redesign, and possibly a new detection angle
entirely.

Raising a re-tuning request for what is actually rule decay burns a quarter
and leaves the risk uncovered.
""")

# %% [markdown]
# ## 6. Monitoring triggers

# %%
banner("6. MONITORING TRIGGERS")

triggers = pd.DataFrame([
    ("Alert volume", f"Outside {limits['lower']:,.0f}-{limits['upper']:,.0f}", "Monthly",
     "Investigate; re-tune if sustained two periods"),
    ("Precision", "Below baseline lower limit", "Monthly", "Investigate label and typology changes"),
    ("PSI on the rule variable", "> 0.25", "Monthly", "Re-tune against the current distribution"),
    ("PSI on the rule variable", "0.10 - 0.25", "Monthly", "Note and watch; no action alone"),
    ("Segment mix", "Any segment share moves > 5pp", "Quarterly", "Re-check segment calibration"),
    ("BTL productive rate", "Above the tuning-paper upper bound", "Annually", "Full re-tune"),
], columns=["Metric", "Trigger", "Frequency", "Action"])
show(triggers, "A monitoring specification worth signing")

print("""
Every trigger names an ACTION. A threshold with no action attached is a
number in a pack that gets noted and moved past for four quarters running,
and its existence is then cited as evidence the rule was being monitored.
""")

banner("END OF WEEK 7")
print("""
Carry forward into Week 8:
  * Pooled figures hide drift by construction. Always go periodic.
  * Anchor control limits to the baseline, or they chase the drift.
  * PSI + volume + precision, read together, name the failure mode.
  * Population drift and rule decay need opposite remedies.
""")
