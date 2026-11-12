# %% [markdown]
# # Week 5 -- Segment-Based Calibration
#
# **Concept:** One threshold rarely fits all customer types.
#
# **Topics**
#
# * Retail versus Corporate
# * Geography segmentation
# * Product segmentation
# * Peer groups
#
# **Synthetic Dataset**
#
# ```python
# df["segment"] = np.where(
#     np.random.rand(len(df)) > 0.7,
#     "Corporate",
#     "Retail"
# )
# ```
#
# **Exercise.** Compare a single threshold against segment-specific thresholds.
#
# Example: Retail = £20k, Corporate = £100k
#
# **Success criteria.** Explain the improvement achieved without creating
# unjustified complexity.

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

from tmtuning import (answer, banner, apply_rule, calibrate_segments, classification_metrics,
                      compare_uniform_vs_segmented, generate_population, segment_summary,
                      segment_sweeps, show, uniform_threshold_at_capacity)
from tmtuning.plots import plot_segment_curves, save

OUT = _ROOT / "outputs"
CAPACITY = 2_500

# %% [markdown]
# ## 1. The exercise as set: Retail versus Corporate

# %%
banner("1. THE SPEC EXERCISE -- TWO SEGMENTS, TWO THRESHOLDS")

population = generate_population(n=40_000, seed=505, n_periods=12)

# The spec's segment assignment, applied to our population.
rng = np.random.default_rng(505)
spec_df = population.copy()
spec_df["spec_segment"] = np.where(rng.random(len(spec_df)) > 0.7, "Corporate", "Retail")

show(spec_df.groupby("spec_segment").agg(
        rows=("case", "size"), cases=("case", "sum"), prevalence=("case", "mean"),
        median_amount=("monthly_wire_value", "median")).reset_index(),
     "Segments assigned at random, as the spec's np.where does")

print("""
Read those two rows carefully before going further. The segments have
near-identical base rates and near-identical median amounts -- because
`np.random.rand() > 0.7` assigns the label at random, with no reference to
any customer attribute.

This is the same lesson Week 2 taught about the independent `case` label,
in a different place: a segmentation that does not correlate with behaviour
cannot improve detection, no matter how the thresholds are set. Demonstrated
below, then repeated on segments that mean something.
""")

# %%
SINGLE = 50_000
RETAIL_CUT, CORPORATE_CUT = 20_000, 100_000

single_flags = apply_rule(spec_df["monthly_wire_value"], SINGLE)
segmented_flags = np.where(
    spec_df["spec_segment"] == "Corporate",
    spec_df["monthly_wire_value"] > CORPORATE_CUT,
    spec_df["monthly_wire_value"] > RETAIL_CUT,
)

comparison = pd.DataFrame([
    {"approach": f"Single threshold £{SINGLE:,}",
     **classification_metrics(spec_df["case"], single_flags)},
    {"approach": f"Retail £{RETAIL_CUT:,} / Corporate £{CORPORATE_CUT:,}",
     **classification_metrics(spec_df["case"], segmented_flags)},
])
show(comparison[["approach", "alerts", "tp", "precision", "recall",
                 "alerts_per_true_positive"]],
     "Single threshold versus the spec's segment-specific thresholds")

print("""
The segmented version raises MORE alerts and finds more cases, because a £20k
retail cut is far looser than £50k across a randomly-labelled 70% of the
population. That is a volume change wearing the costume of an improvement.

Compare precision, not counts. If precision is unchanged, the extra cases came
from raising more alerts, and the same result was available by lowering the
single threshold -- without adding a second parameter to govern.
""")

# %%
# The fair test: give the single threshold the same alert budget.
budget = int(np.asarray(segmented_flags).sum())
matched_cut = float(np.quantile(spec_df["monthly_wire_value"], 1 - budget / len(spec_df)))
matched = classification_metrics(spec_df["case"], apply_rule(spec_df["monthly_wire_value"], matched_cut))

fair = pd.DataFrame([
    {"approach": f"Single threshold £{matched_cut:,.0f} (volume-matched)", **matched},
    {"approach": f"Retail £{RETAIL_CUT:,} / Corporate £{CORPORATE_CUT:,}",
     **classification_metrics(spec_df["case"], segmented_flags)},
])
show(fair[["approach", "alerts", "tp", "precision", "recall"]],
     f"Both spending the same {budget:,} alerts")

print("""
At equal alert volume the random segmentation gives back whatever it appeared
to gain. Two thresholds, twice the governance, no detection benefit.

Every segmentation proposal must be tested this way. The comparison that makes
segmentation look good -- segmented rule against an untuned single threshold
firing fewer alerts -- is not a comparison, it is a rigged one.
""")

# %% [markdown]
# ## 2. Segments that mean something
#
# Now repeat on segments defined by what customers actually are, rather than by
# a coin flip.

# %% [markdown]
# ## 2a. Profile the segments first
#
# Do this before building anything. If the segments do not differ materially in
# value scale or base rate, segment thresholds add governance overhead for no
# detection benefit -- and the right recommendation is not to build them.

# %%
banner("2. SEGMENT PROFILE")

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
# ## 3. Per-segment yield curves

# %%
banner("3. PER-SEGMENT YIELD CURVES")

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
# ## 4. The cost of a single global threshold

# %%
banner("4. WHAT ONE GLOBAL THRESHOLD DOES TO EACH SEGMENT")

uniform = uniform_threshold_at_capacity(population, capacity=CAPACITY)
show(uniform, f"A single threshold tuned to {CAPACITY:,} alerts, broken down by segment")

starved = uniform[uniform["alerts"] == 0]["segment"].tolist()
if starved:
    print(f"\nSegments receiving ZERO alerts: {', '.join(starved)}")
    print("Their risk is not being monitored by this rule at all -- while their")
    print("cases still count against measured recall.")

# %% [markdown]
# ## 5. Calibrated allocation at the same budget

# %%
banner("5. SEGMENT-CALIBRATED THRESHOLDS")

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
# ## 6. The headline comparison

# %%
banner("6. UNIFORM VERSUS SEGMENTED, SAME BUDGET")

segment_comparison = compare_uniform_vs_segmented(population, capacity=CAPACITY)
show(segment_comparison[["segment", "threshold_uniform", "alerts_uniform", "tp_uniform",
                 "threshold_segmented", "alerts_segmented", "tp_segmented",
                 "uplift_tp", "alert_delta"]],
     "Both options costed at the same total alert budget")

total = segment_comparison[segment_comparison["segment"] == "TOTAL"].iloc[0]
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
banner("7. SENSITIVITY: DOES THE UPLIFT HOLD AT OTHER BUDGETS?")

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
  * A segmentation uncorrelated with behaviour cannot improve detection.
  * Profile before you segment. 'No uplift' is a legitimate finding.
  * Compare at EQUAL alert budgets or the comparison is rigged.
  * More thresholds means more governance and a new data-quality control.
  * Week 6 changes the rule's LOGIC rather than its thresholds.
""")
