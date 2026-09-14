# %% [markdown]
# # Week 5 -- Segment-Specific Calibration
#
# **Concept:** One threshold across a heterogeneous portfolio is a compromise
# that serves no segment well. Quantify the cost of that compromise and
# allocate a fixed alert budget where it detects the most.
#
# **Topics**
#
# * Segment profiling -- when segmentation is and is not justified
# * Per-segment yield curves
# * Allocating a fixed alert budget across segments
# * The governance cost of more thresholds
#
# **Success criteria.** Show the detection uplift from segment calibration at
# an unchanged total alert volume -- or show that there isn't one.

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

from tmtuning import (answer, banner, calibrate_segments, compare_uniform_vs_segmented,
                      generate_population, segment_summary, segment_sweeps, show,
                      uniform_threshold_at_capacity)
from tmtuning.plots import plot_segment_curves, save

OUT = _ROOT / "outputs"
CAPACITY = 2_500

# %% [markdown]
# ## 1. Profile the segments first
#
# Do this before building anything. If the segments do not differ materially in
# value scale or base rate, segment thresholds add governance overhead for no
# detection benefit -- and the right recommendation is not to build them.

# %%
banner("1. SEGMENT PROFILE")

population = generate_population(n=40_000, seed=505, n_periods=12)
profile = segment_summary(population)
show(profile.reset_index(), "Segment profile, sorted by case concentration")

spread = profile["value_median"].max() / profile["value_median"].min()
rate_spread = profile["prevalence"].max() / profile["prevalence"].min()
print(f"\nMedian transaction value spans {spread:.0f}x across segments.")
print(f"Base rate spans {rate_spread:.0f}x across segments.")
print("""
`case_concentration` is the share of all cases a segment holds divided by its
share of rows. Above 1 means the segment carries more risk than its size
implies; below 1 means less.

Both spreads are large here, so segmentation has something to work with. If
both were close to 1x, the honest recommendation would be a single global
threshold and a note explaining why segmentation was considered and rejected.
""")

# %% [markdown]
# ## 2. Per-segment yield curves

# %%
banner("2. PER-SEGMENT YIELD CURVES")

sweeps = segment_sweeps(population, "monthly_wire_value", "case", "segment", n_thresholds=40)
save(plot_segment_curves(sweeps, title="Yield curves by segment"), OUT / "week05_segment_curves.png")
print(f"Chart written to {OUT / 'week05_segment_curves.png'}")

for name, sweep in sorted(sweeps.items()):
    mid = sweep.iloc[len(sweep) // 2]
    print(f"  {name:<10} at its median candidate £{mid['threshold']:>10,.0f}: "
          f"{int(mid['alerts']):>5,} alerts, precision {mid['precision']:.2%}")

print("\nEach segment's grid is built from its OWN distribution. A grid derived")
print("from the pooled population would be dominated by the largest segment and")
print("would barely sample the range where a corporate decision actually lies.")

# %% [markdown]
# ## 3. The cost of a single global threshold

# %%
banner("3. WHAT ONE GLOBAL THRESHOLD DOES TO EACH SEGMENT")

uniform = uniform_threshold_at_capacity(population, capacity=CAPACITY)
show(uniform, f"A single threshold tuned to {CAPACITY:,} alerts, broken down by segment")

starved = uniform[uniform["alerts"] == 0]["segment"].tolist()
if starved:
    print(f"\nSegments receiving ZERO alerts: {', '.join(starved)}")
    print("Their risk is not being monitored by this rule at all -- while their")
    print("cases still count against measured recall.")

# %% [markdown]
# ## 4. Calibrated allocation at the same budget

# %%
banner("4. SEGMENT-CALIBRATED THRESHOLDS")

calibrated = calibrate_segments(population, total_capacity=CAPACITY)
show(calibrated, f"Greedy marginal allocation of the same {CAPACITY:,} alerts")

print("""
The allocation rule: spend each next alert wherever that alert is most likely
to be productive. That is what a triage manager does by instinct; this makes
it explicit, reproducible and auditable.

One caveat to carry into the paper: greedy allocation is provably optimal only
when each segment's marginal yield falls monotonically as its threshold
loosens. Real sweeps are noisy and can briefly violate that, so this is a
near-optimal allocation, not a certified optimum. Do not claim optimality you
have not verified.
""")

# %% [markdown]
# ## 5. The headline comparison

# %%
banner("5. UNIFORM VERSUS SEGMENTED, SAME BUDGET")

comparison = compare_uniform_vs_segmented(population, capacity=CAPACITY)
show(comparison[["segment", "threshold_uniform", "alerts_uniform", "tp_uniform",
                 "threshold_segmented", "alerts_segmented", "tp_segmented",
                 "uplift_tp", "alert_delta"]],
     "Both options costed at the same total alert budget")

total = comparison[comparison["segment"] == "TOTAL"].iloc[0]
uplift_pct = total["uplift_tp"] / max(total["tp_uniform"], 1)
print(f"\nDetection uplift: {int(total['uplift_tp']):+,} cases "
      f"({uplift_pct:+.1%}) for {int(total['alert_delta']):+,} alerts.")
print(f"Recall: {total['recall_uniform']:.2%} -> {total['recall_segmented']:.2%}")

# %%
answer("Is this uplift worth having?",
       f"""
It is {uplift_pct:+.1%} more detection for no additional investigator effort,
which is real. But the decision is not arithmetic alone -- weigh it against
what more thresholds cost you:

  * Every threshold is a parameter that must be documented, approved,
    monitored and re-tuned. Four thresholds is four times the tuning work
    and four times the drift surface.
  * Segment assignment becomes control-relevant. If a customer can be
    mis-segmented, they can be under-monitored -- and segment data quality
    is now an AML control, with everything that implies.
  * Segment definitions must be stable. If Relationship Management can
    re-tier a customer, they can move them into a looser threshold without
    anyone treating it as a monitoring change.

The recommendation writes itself only when the uplift is large. When it is
marginal, the defensible answer is often to segment into two groups rather
than four, capturing most of the benefit for a fraction of the overhead.
""")

# %%
banner("6. SENSITIVITY: DOES THE UPLIFT HOLD AT OTHER BUDGETS?")

rows = []
for capacity in [1_000, 2_000, 2_500, 4_000, 6_000]:
    total_row = compare_uniform_vs_segmented(population, capacity=capacity)
    total_row = total_row[total_row["segment"] == "TOTAL"].iloc[0]
    rows.append({
        "capacity": capacity,
        "tp_uniform": int(total_row["tp_uniform"]),
        "tp_segmented": int(total_row["tp_segmented"]),
        "uplift_tp": int(total_row["uplift_tp"]),
        "uplift_pct": total_row["uplift_tp"] / max(total_row["tp_uniform"], 1),
    })
show(pd.DataFrame(rows), "Uplift across alert budgets")
print("\nA finding that only holds at one budget is a coincidence. Check the")
print("shape before recommending -- and report the budget your recommendation")
print("assumes, because that is the number most likely to change on you.")

banner("END OF WEEK 5")
print("""
Carry forward into Week 6:
  * Profile before you segment. 'No uplift' is a legitimate finding.
  * Compare at EQUAL alert budgets or the comparison is rigged.
  * More thresholds means more governance and a new data-quality control.
  * Everything so far measures risk we can see. Week 6 measures what we miss.
""")
