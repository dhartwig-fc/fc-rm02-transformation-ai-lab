# %% [markdown]
# # Week 8 -- Advanced Calibration Using Scoring
#
# **Concept:** Move beyond a single threshold.
#
# **Topics**
#
# * Risk scoring
# * Weighted indicators
# * Probability estimates
# * Score cut-offs
#
# **Scoring Rule**
#
# ```python
# score = (
#     amount_score * 0.5 +
#     velocity_score * 0.3 +
#     geo_score * 0.2
# )
# ```
#
# **Exercise.** Create deciles, score distributions and yield by decile.
#
# **Deliverable.** Determine the optimal score cut-off.
#
# **Success criteria.** Justify the chosen cut-off using evidence rather than
# intuition.

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

from tmtuning import (answer, banner, apply_rule, band_score, calibrate_probability,
                      classification_metrics, decile_table, generate_population,
                      optimise_cutoff, percentile_score, score_cutoff_sweep, show,
                      split_by_period, weighted_score)
from tmtuning.plots import PALETTE, save

OUT = _ROOT / "outputs"
CURRENT_THRESHOLD = 50_000
CAPACITY = 2_000

# %% [markdown]
# ## 1. Build the component scores
#
# Before weighting anything, the components have to be made commensurable.
# Pounds and transaction counts cannot be added; their percentile ranks can.

# %%
banner("1. COMPONENT SCORES")

population = generate_population(n=60_000, seed=808, n_periods=12)
in_time, out_of_time = split_by_period(population, holdout_periods=3)

#: Jurisdiction tiers are an ordered judgement, not a measurement, so the geo
#: component is an explicit lookup rather than a rank. Someone chose these
#: numbers and has to defend them -- which is the correct state of affairs.
GEO_BANDS = {"DOMESTIC": 0, "STANDARD": 40, "ELEVATED": 75, "HIGH": 100}


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the three component scores and the weighted total."""
    out = df.copy()
    out["amount_score"] = percentile_score(out["monthly_wire_value"])
    out["velocity_score"] = percentile_score(out["velocity"])
    out["geo_score"] = band_score(out["jurisdiction_risk"], GEO_BANDS)
    out["score"] = weighted_score({
        "amount": (out["amount_score"], 0.5),
        "velocity": (out["velocity_score"], 0.3),
        "geo": (out["geo_score"], 0.2),
    })
    return out


in_time = add_scores(in_time)
out_of_time = add_scores(out_of_time)

show(in_time[["monthly_wire_value", "amount_score", "velocity", "velocity_score",
              "jurisdiction_risk", "geo_score", "score", "case"]].head(6),
     "Components and the weighted score")

print("""
Percentile ranks, not raw values. Two reasons, and both bite in practice:

  * Raw pounds would swamp the sum. A £200,000 amount alongside a velocity of
    7 makes the weights decorative -- amount would decide every alert whatever
    number sat next to it.
  * Ranks take their average on ties, which matters for a discrete count.
    Without it every customer with velocity 0 gets an arbitrary ordering among
    themselves, and the component carries precision the data does not support.
""")

# %% [markdown]
# ## 2. Weighted indicators
#
# The weights are the model. They should be defensible individually.

# %%
banner("2. WEIGHTED INDICATORS")

weights = pd.DataFrame([
    ("amount_score", 0.5, "Percentile rank of monthly outbound wire value",
     "Carries the most weight because it is the incumbent rule's only input, "
     "so the score must not regress against it on value alone."),
    ("velocity_score", 0.3, "Percentile rank of outbound wire transaction count",
     "Adds the structuring/layering angle the amount rule is blind to."),
    ("geo_score", 0.2, "Jurisdiction tier, mapped 0/40/75/100",
     "Lowest weight because it is a static customer attribute, not behaviour, "
     "and it changes only at periodic review."),
], columns=["Component", "Weight", "Definition", "Rationale"])
show(weights, "Every weight needs a sentence someone will defend")

print("""
Weights that do not sum to 1 silently rescale the score, so a cut-off of 60
means something different from one run to the next. `weighted_score` refuses
them by default rather than letting that through quietly.

Note what the rationale column is NOT: "0.5 performed best". Choosing weights
by performance on the same data the performance is measured on is circular,
and it is the first thing independent challenge will find (Week 9).
""")

# %%
# Does each component earn its place? Score each one alone, at matched volume.
banner("2b. DOES EACH COMPONENT EARN ITS WEIGHT?")

incumbent_flags = apply_rule(in_time["monthly_wire_value"], CURRENT_THRESHOLD)
target_alerts = int(np.asarray(incumbent_flags).sum())

rows = []
for name in ["amount_score", "velocity_score", "geo_score", "score"]:
    cut = float(np.quantile(in_time[name], 1 - target_alerts / len(in_time)))
    metrics = classification_metrics(in_time["case"], in_time[name] >= cut)
    rows.append({"scored_on": name, "cutoff": cut, **metrics})
rows.append({"scored_on": f"incumbent > £{CURRENT_THRESHOLD:,}", "cutoff": CURRENT_THRESHOLD,
             **classification_metrics(in_time["case"], incumbent_flags)})
show(pd.DataFrame(rows)[["scored_on", "cutoff", "alerts", "tp", "precision", "recall"]],
     f"Each component alone, all cut to ~{target_alerts:,} alerts")

print("""
`geo_score` alone is weak -- it is a four-value lookup, so it cannot sort
within a tier and its 'cut-off' lands arbitrarily. That is not an argument for
dropping it: a component can be poor alone and still improve a blend, by
separating records the other components rank identically. What it does argue
against is giving it a large weight.
""")

# %% [markdown]
# ## 3. Score distribution

# %%
banner("3. SCORE DISTRIBUTION")

distribution = in_time.groupby(pd.cut(in_time["score"], bins=10, precision=0), observed=True).agg(
    records=("case", "size"), cases=("case", "sum"), case_rate=("case", "mean"))
distribution["share_of_population"] = distribution["records"] / len(in_time)
show(distribution.reset_index().rename(columns={"score": "score_band"}),
     "Records and case rate by score band")

print(f"""
Score range {in_time['score'].min():.1f} to {in_time['score'].max():.1f}, median {in_time['score'].median():.1f}.

The distribution is roughly symmetric because two of three components are
percentile ranks, which are uniform by construction. That is a property of how
the score was built, not a finding about the portfolio -- do not present it as
evidence the score is well behaved.
""")

# %% [markdown]
# ## 4. Yield by decile
#
# The single most informative table a score produces.

# %%
banner("4. YIELD BY DECILE")

deciles = decile_table(in_time, "score", "case")
show(deciles[["decile", "records", "cases", "min_score", "max_score", "case_rate", "lift",
              "cumulative_alert_rate", "cumulative_recall", "cumulative_precision"]],
     "Decile 1 is the highest-scoring 10%")

top = deciles.iloc[0]
print(f"""
Decile 1 holds {top['case_rate']:.1%} case rate against a {in_time['case'].mean():.2%} base rate -- {top['lift']:.1f}x lift --
and captures {top['cumulative_recall']:.0%} of all cases in {top['cumulative_alert_rate']:.0%} of the population.

A steep, monotone gradient down the table is what a working score looks like.
A flat table means the score is not separating risk, however good its headline
precision appears -- and a table that is monotone except for one decile is
usually a small-sample wobble rather than a finding.
""")

# %%
# Chart: yield by decile, with the base rate as the reference.
import matplotlib.pyplot as plt

from tmtuning.plots import INK, _style

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(deciles["decile"], deciles["case_rate"], color=PALETTE[0], zorder=3, width=0.7)
base_rate = in_time["case"].mean()
ax.axhline(base_rate, color=INK["muted"], linewidth=1.4, linestyle=":", zorder=4)
ax.text(10.4, base_rate, f" base rate {base_rate:.1%}", color=INK["secondary"],
        fontsize=9, va="bottom", ha="right")
for _, row in deciles.iterrows():
    ax.annotate(f"{row['case_rate']:.1%}", (row["decile"], row["case_rate"]),
                textcoords="offset points", xytext=(0, 4), ha="center",
                color=INK["secondary"], fontsize=8)
ax.set_xticks(deciles["decile"])
_style(ax, "Case rate by score decile", "Score decile (1 = highest scoring)", "Case rate",
       subtitle="A steep, monotone gradient is what a working score looks like")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
fig.tight_layout()
save(fig, OUT / "week08_yield_by_decile.png")
print(f"Chart written to {OUT / 'week08_yield_by_decile.png'}")

# %% [markdown]
# ## 5. Probability estimates
#
# "Score 72" means nothing on its own. Convert it to something an investigator
# and a validator can both act on.

# %%
banner("5. PROBABILITY ESTIMATES")

calibration = calibrate_probability(in_time, "score", "case", n_bins=10)
show(calibration[["band", "records", "cases", "observed_probability", "ci_low", "ci_high"]],
     "Observed case probability by score band, with 95% Wilson intervals")

print("""
Read the confidence intervals, not just the point estimates. At the top of the
range the bands hold few records, so their intervals are wide -- often wider
than the gap to the neighbouring band. A calibration table that looks neatly
monotone can be entirely consistent with a flat one up there.

This is what "probability estimates" has to mean in a tuning paper: a rate with
an interval attached, from a band with enough records to support it. Quoting
"records scoring 90+ are 22% likely to be cases" off 40 records is a number
that will not reproduce next quarter.
""")

# %% [markdown]
# ## 6. Score cut-offs

# %%
banner("6. SCORE CUT-OFFS")

cutoffs = np.arange(30, 96, 5)
sweep = score_cutoff_sweep(in_time, cutoffs, "score", "case")
show(sweep[["cutoff", "alerts", "tp", "fn", "precision", "recall",
            "alerts_per_true_positive"]],
     "Metrics at each candidate cut-off")

chosen = optimise_cutoff(sweep, objective="recall", max_alerts=CAPACITY)
print(f"\n  Capacity constraint: {CAPACITY:,} alerts")
print(f"  Chosen cut-off:      {chosen['cutoff']:.0f}")
print(f"    alerts     {int(chosen['alerts']):,}")
print(f"    precision  {chosen['precision']:.2%}")
print(f"    recall     {chosen['recall']:.2%}")

# %%
# The comparison that matters: score versus the incumbent single threshold,
# at matched alert volume.
banner("6b. SCORE VERSUS SINGLE THRESHOLD, MATCHED VOLUME")

incumbent_metrics = classification_metrics(in_time["case"], incumbent_flags)
matched_cut = float(np.quantile(in_time["score"], 1 - target_alerts / len(in_time)))
matched_metrics = classification_metrics(in_time["case"], in_time["score"] >= matched_cut)

comparison = pd.DataFrame([
    {"approach": f"Single threshold > £{CURRENT_THRESHOLD:,}", **incumbent_metrics},
    {"approach": f"Weighted score >= {matched_cut:.1f}", **matched_metrics},
])
show(comparison[["approach", "alerts", "tp", "fn", "precision", "recall",
                 "alerts_per_true_positive"]],
     f"Both spending ~{target_alerts:,} alerts")

uplift = matched_metrics["tp"] - incumbent_metrics["tp"]
print(f"\n  Detection uplift at equal alert volume: {uplift:+,} cases "
      f"({uplift / max(incumbent_metrics['tp'], 1):+.1%})")
print("""
Matched volume is the only fair test. A score set to fire more often will of
course find more cases; the question is whether it finds more for the same
investigator effort.
""")

# %% [markdown]
# ## 7. Does the cut-off hold out of time?

# %%
banner("7. OUT-OF-TIME CONFIRMATION")

cut = float(chosen["cutoff"])
oot = pd.DataFrame([
    {"sample": "In-time", **classification_metrics(in_time["case"], in_time["score"] >= cut)},
    {"sample": "Out-of-time",
     **classification_metrics(out_of_time["case"], out_of_time["score"] >= cut)},
])
show(oot[["sample", "alerts", "tp", "precision", "recall", "alert_rate"]],
     f"Cut-off {cut:.0f} applied to both samples (compare RATES, not counts)")

change = (oot.loc[1, "precision"] - oot.loc[0, "precision"]) * 100
print(f"\n  Precision change out of time: {change:+.2f}pp")
print("""
One trap specific to scores. The component scores are percentile ranks computed
WITHIN each sample, so a rank of 80 means "top 20% of this sample" -- not a
fixed pound value. That makes the alert rate stable by construction and hides
drift that a fixed threshold would expose.

In production, fix the rank boundaries from the tuning window and apply those
same boundaries to later periods. Otherwise the score silently re-tunes itself
every period, and Week 7's drift monitoring has nothing to detect.
""")

# %% [markdown]
# ## 8. Deliverable: the optimal cut-off, justified by evidence

# %%
banner("8. DELIVERABLE -- CUT-OFF JUSTIFICATION")

at_cut = deciles[deciles["max_score"] >= cut]
deciles_covered = len(at_cut)

print(f"""  SCORING RULE

    score = amount_score * 0.5 + velocity_score * 0.3 + geo_score * 0.2

  PROPOSED CUT-OFF: {cut:.0f}

  EVIDENCE

  1. Capacity. The cut-off produces {int(chosen['alerts']):,} alerts against a
     {CAPACITY:,} constraint. Chosen to maximise recall subject to that
     constraint, not to maximise a composite metric.

  2. Yield gradient. It falls in decile {deciles_covered}, where the case rate is
     {at_cut.iloc[-1]['case_rate']:.2%} against a {in_time['case'].mean():.2%} base rate
     ({at_cut.iloc[-1]['lift']:.1f}x lift). The deciles below it run close to the
     base rate, so alerts there are near-random.

  3. Matched-volume comparison. At the incumbent's alert volume the score finds
     {uplift:+,} more cases ({uplift / max(incumbent_metrics['tp'], 1):+.1%}) for the same effort.

  4. Probability. Records at or above this cut-off were cases
     {chosen['precision']:.1%} of the time, with the interval in section 5.

  5. Out of time. Precision moves {change:+.2f}pp on periods held back from
     tuning, so the cut-off is not fitted to the window.

  WHAT THIS ACCEPTS

  {int(chosen['fn']):,} cases fall below the cut-off and are not alerted. The
  score does not reduce that number relative to a threshold at the same volume;
  it changes WHICH cases are missed -- away from high-value low-velocity
  customers and towards low-value ones in domestic jurisdictions. That
  re-allocation is a risk-appetite decision, not a technical improvement, and
  it belongs in front of the risk owner explicitly.
""")

answer("Why is 'justified by evidence' harder for a score than for a threshold?",
       """
Because a score has more places to hide a judgement.

A threshold has one number, and a sweep shows exactly what every alternative
would have done. A score has the component definitions, the weights, the
combination method AND the cut-off -- four sets of choices, of which only the
last one shows up in a sweep.

So the evidence has to cover all four:

  * why each component is in the score, and its lift on its own;
  * why each weight is what it is, from a rationale that does not reduce to
    "it performed best on this data";
  * why they are combined additively rather than as a product or a decision
    tree;
  * and only then, why this cut-off.

A tuning paper that sweeps the cut-off and says nothing about the weights has
justified one of the four choices and presented it as if it were all of them.
That is the specific failure independent challenge looks for in scoring models
(Week 9).
""")

banner("END OF WEEK 8")
print("""
Carry forward into Week 9:
  * Rank components before weighting, or the largest unit wins by default.
  * Weights are the model. Each needs a rationale, not a fit statistic.
  * Deciles and lift show whether a score separates risk. Nothing else does it
    as directly.
  * Probability estimates need confidence intervals, or they will not reproduce.
  * Fix rank boundaries from the tuning window, or the score re-tunes itself.
""")
