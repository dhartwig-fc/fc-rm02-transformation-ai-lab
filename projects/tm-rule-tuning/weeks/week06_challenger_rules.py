# %% [markdown]
# # Week 6 -- Challenger Rule Development
#
# **Concept:** Develop alternative logic to challenge current production
# settings.
#
# **Topics**
#
# * Incumbent versus challenger
# * Risk indicators
# * Composite scores
# * Explainability
#
# **Scenario**
#
# Current rule:
#
# > Transaction amount > £50k
#
# Challenger:
#
# > Amount > £30k **AND** Velocity > 5 transactions
#
# **Python Exercise.** Generate `velocity`, `amount` and `customer_risk`.
# Compare both rules.
#
# **Metrics.** Evaluate precision, recall and volume.
#
# **Success criteria.** Determine whether the challenger merits further
# validation.

# %%
# --- path bootstrap ---
import pathlib
import sys

for _p in pathlib.Path(__file__ if "__file__" in globals() else "x").resolve().parents:
    if (_p / "src" / "tmtuning").is_dir():
        sys.path.insert(0, str(_p / "src"))
        break

import numpy as np
import pandas as pd

from tmtuning import (answer, banner, apply_rule, classification_metrics, compare_rules,
                      generate_population, rule_overlap, show, split_by_period)

# %% [markdown]
# ## 1. Generate velocity, amount and customer risk

# %%
banner("1. THE POPULATION")

population = generate_population(n=60_000, seed=606, n_periods=12)
in_time, out_of_time = split_by_period(population, holdout_periods=3)

show(population[["segment", "monthly_wire_value", "velocity", "customer_risk",
                 "high_risk_jurisdiction", "case"]].head(6),
     "amount (monthly_wire_value), velocity and customer_risk")

print(f"\n  Population {len(population):,} customer-months, "
      f"{population['case'].sum():,} cases ({population['case'].mean():.2%})")

# %% [markdown]
# ## 2. Check the risk indicators before building on them
#
# A challenger is only as good as the indicators it adds. Test each one on its
# own first: an indicator that does not separate risk will not start doing so
# because it has been ANDed to something that does.

# %%
banner("2. RISK INDICATORS")

indicators = pd.DataFrame([
    {"indicator": "velocity > 5",
     "flagged": int((population["velocity"] > 5).sum()),
     "case_rate_flagged": population.loc[population["velocity"] > 5, "case"].mean(),
     "case_rate_not": population.loc[population["velocity"] <= 5, "case"].mean()},
    {"indicator": "customer_risk == HIGH",
     "flagged": int((population["customer_risk"] == "HIGH").sum()),
     "case_rate_flagged": population.loc[population["customer_risk"] == "HIGH", "case"].mean(),
     "case_rate_not": population.loc[population["customer_risk"] != "HIGH", "case"].mean()},
    {"indicator": "high_risk_jurisdiction",
     "flagged": int(population["high_risk_jurisdiction"].sum()),
     "case_rate_flagged": population.loc[population["high_risk_jurisdiction"] == 1, "case"].mean(),
     "case_rate_not": population.loc[population["high_risk_jurisdiction"] == 0, "case"].mean()},
    {"indicator": "amount > £50k",
     "flagged": int((population["monthly_wire_value"] > 50_000).sum()),
     "case_rate_flagged": population.loc[population["monthly_wire_value"] > 50_000, "case"].mean(),
     "case_rate_not": population.loc[population["monthly_wire_value"] <= 50_000, "case"].mean()},
])
indicators["lift"] = indicators["case_rate_flagged"] / indicators["case_rate_not"]
indicators["coverage"] = indicators["flagged"] / len(population)
show(indicators, "Each indicator on its own")

print("""
`lift` is how many times more likely a flagged record is to be a case. `coverage`
is how much of the population it flags. A good challenger ingredient has decent
lift AND enough coverage to matter -- an indicator with lift of 10 that fires on
0.1% of customers cannot move a rule's recall no matter how it is combined.
""")

# %% [markdown]
# ## 3. The scenario: incumbent versus challenger

# %%
banner("3. INCUMBENT VERSUS CHALLENGER")


def build_rules(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Define every candidate once, so identical logic is scored on both samples."""
    amount = df["monthly_wire_value"]
    velocity = df["velocity"]
    return {
        # Current production rule.
        "incumbent: amount > 50k": np.asarray(apply_rule(amount, 50_000)),
        # The challenger as specified.
        "challenger: amount > 30k AND velocity > 5": np.asarray((amount > 30_000) & (velocity > 5)),
        # Two variants, to show the AND/OR choice is the whole design.
        "variant: amount > 30k OR velocity > 5": np.asarray((amount > 30_000) | (velocity > 5)),
        "variant: amount > 30k AND (velocity > 5 OR risk HIGH)":
            np.asarray((amount > 30_000) & ((velocity > 5) | (df["customer_risk"] == "HIGH"))),
    }


rules = build_rules(in_time)
comparison = compare_rules(in_time, rules, champion="incumbent: amount > 50k")
show(comparison[["alerts", "tp", "fn", "precision", "recall",
                 "alerts_per_true_positive", "delta_tp", "delta_alerts"]].reset_index(),
     "All four scored on the same sample, with the same labels")

print("""
The specified challenger is the second row. Note what the AND does: it more
than halves alert volume against the incumbent while improving precision
sharply, but recall falls -- because a customer moving £2m in three large
transfers fails the velocity test and is no longer alerted at all.

The OR variant is the mirror image: high recall, poor precision, unworkable
volume. Same two indicators, opposite operating point. The combination
operator, not the thresholds, is doing most of the work here.
""")

# %% [markdown]
# ## 4. Break the netting open

# %%
banner("4. ALERT OVERLAP -- WHAT NETTING HIDES")

champion_flags = rules["incumbent: amount > 50k"]
challenger_flags = rules["challenger: amount > 30k AND velocity > 5"]

overlap = rule_overlap(in_time, champion_flags, challenger_flags)
show(overlap, "Incumbent vs the specified challenger")

champion_only = overlap.loc[overlap["cell"] == "Champion only", "cases"].iloc[0]
challenger_only = overlap.loc[overlap["cell"] == "Challenger only", "cases"].iloc[0]
print(f"\n  Newly missed by the challenger : {champion_only:,} cases")
print(f"  Newly caught by the challenger : {challenger_only:,} cases")
print(f"  Net                            : {challenger_only - champion_only:+,} cases")

# %%
missed = in_time[np.asarray(champion_flags) & ~np.asarray(challenger_flags) & (in_time["case"] == 1)]
show(missed.groupby("segment").agg(
        cases=("case", "size"),
        median_amount=("monthly_wire_value", "median"),
        median_velocity=("velocity", "median")).reset_index(),
     "Who the challenger stops catching")

print("""
This is the table that decides the recommendation, and the netted figure in
section 3 cannot show it. The newly-missed cases are high-value, low-velocity
customers -- which is a coherent typology, not a random sample of the
incumbent's catch. A challenger that systematically drops one typology has
narrowed the bank's coverage, whatever its precision does.
""")

# %% [markdown]
# ## 5. Composite scores

# %%
banner("5. COMPOSITE SCORES -- A PREVIEW")

print("""
A binary rule forces every indicator to a hard cut. A composite score keeps the
gradient: a customer just under the amount cut but well over on velocity, and
sitting in a high-risk jurisdiction, can still surface.

Here it is in outline, to show what the AND is throwing away. Week 8 builds it
properly -- weighted components, deciles, yield by decile and a justified
cut-off.
""")


def preview_score(df: pd.DataFrame) -> np.ndarray:
    """Points for each risk indicator. Deliberately round numbers a person chose."""
    amount_points = np.select(
        [df["monthly_wire_value"] > 100_000, df["monthly_wire_value"] > 50_000,
         df["monthly_wire_value"] > 30_000], [3, 2, 1], default=0)
    velocity_points = np.select(
        [df["velocity"] > 8, df["velocity"] > 5, df["velocity"] > 3], [3, 2, 1], default=0)
    risk_points = np.select(
        [df["customer_risk"] == "HIGH", df["customer_risk"] == "MEDIUM"], [2, 1], default=0)
    return amount_points + velocity_points + risk_points + 2 * df["high_risk_jurisdiction"].to_numpy()


in_time_scored = in_time.assign(score=preview_score(in_time))
incumbent_alerts = int(np.asarray(champion_flags).sum())

rows = []
for cut in range(3, 10):
    rows.append({"score_cutoff": cut,
                 **classification_metrics(in_time["case"], in_time_scored["score"] >= cut)})
score_sweep = pd.DataFrame(rows)

matched = score_sweep.iloc[(score_sweep["alerts"] - incumbent_alerts).abs().argmin()]
incumbent_metrics = classification_metrics(in_time["case"], champion_flags)
print(f"  At matched alert volume (~{incumbent_alerts:,} alerts):")
print(f"    incumbent        precision {incumbent_metrics['precision']:.2%}  "
      f"recall {incumbent_metrics['recall']:.2%}")
print(f"    composite (>={int(matched['score_cutoff'])})  precision {matched['precision']:.2%}  "
      f"recall {matched['recall']:.2%}  ({int(matched['alerts']):,} alerts)")
print("""
  Matched volume is the only fair test. A score set to fire more often will of
  course find more cases; the question is whether it finds more for the same
  investigator effort. Week 8 does this properly.
""")

# %% [markdown]
# ## 6. Explainability

# %%
banner("6. EXPLAINABILITY")

example = in_time_scored[in_time_scored["score"] >= 6].iloc[0]
print("  What an investigator sees when a composite score alerts:\n")
print(f"    Customer {example['customer_id']}  ({example['segment']})   SCORE {int(example['score'])}")
print(f"      amount £{example['monthly_wire_value']:,.0f}  -> "
      f"{np.select([example['monthly_wire_value'] > 100_000, example['monthly_wire_value'] > 50_000, example['monthly_wire_value'] > 30_000], [3, 2, 1], default=0)} points")
print(f"      velocity {example['velocity']}            -> "
      f"{np.select([example['velocity'] > 8, example['velocity'] > 5, example['velocity'] > 3], [3, 2, 1], default=0)} points")
print(f"      customer risk {example['customer_risk']:<6}  -> "
      f"{np.select([example['customer_risk'] == 'HIGH', example['customer_risk'] == 'MEDIUM'], [2, 1], default=0)} points")
print(f"      high-risk jurisdiction {example['high_risk_jurisdiction']} -> "
      f"{2 * example['high_risk_jurisdiction']} points")

print("""
  That decomposition is the point. The investigator opens the alert already
  knowing which behaviour triggered it and where to start.

  Contrast "this customer scored 0.83 on the model". Same alert, no starting
  point -- and no way for the investigator to tell the difference between a
  genuine pattern and a data quality artefact upstream.
""")

trade_offs = pd.DataFrame([
    ("Single threshold", "Trivial", "Low", "Trivial", "Narrow -- one behaviour only"),
    ("AND of two conditions", "Easy", "Medium", "Easy", "Narrow, but sharper"),
    ("OR of two conditions", "Easy", "Low", "Easy", "Broad, usually too broad"),
    ("Points-based composite", "Moderate", "Higher", "Moderate", "Broad and graded"),
    ("Fitted model score", "Hard", "Highest", "Hard", "Broadest, least auditable"),
], columns=["Rule type", "Explaining one alert", "Typical precision",
            "Explaining the rule to a regulator", "Coverage"])
show(trade_offs, "The explainability gradient")

answer("Should the bank adopt the composite score?",
       """
Only with the implementation conditions attached, and they are not
formalities.

In its favour: at matched alert volume it detects more, because it keeps the
gradient that a hard AND throws away, and it recovers exactly the high-value
low-velocity cases the specified challenger drops.

Against it, and these decide the answer:

  * Every input becomes an AML control. The score depends on customer_risk,
    which is a KYC rating maintained by another team on another cycle. If a
    rating is stale or defaulted, the score silently under-fires, and it
    under-fires on the customers the rating was meant to flag.

  * Weights need a rationale and a re-tuning cycle. "3 points for amount over
    £100k" must trace to something, and a validator will ask to what.

  * Coverage of the underlying typology has to be re-argued. The rule no
    longer maps to one stated behaviour, so the design rationale that
    justified it stops applying.

The defensible route is usually to run it as a challenger in parallel for a
period and compare real dispositions on the disjoint alerts -- which settles
with investigation outcomes what a backtest can only estimate from proxy
labels.
""")

# %% [markdown]
# ## 7. Out-of-time confirmation

# %%
banner("7. OUT-OF-TIME CONFIRMATION")

oot = compare_rules(out_of_time, build_rules(out_of_time), champion="incumbent: amount > 50k")

# Rates, not counts: the holdout spans fewer periods, so every rule's raw count
# is lower by construction and comparing them says nothing.
stability = pd.DataFrame({
    "precision_in_time": comparison["precision"],
    "precision_out_of_time": oot["precision"],
    "recall_in_time": comparison["recall"],
    "recall_out_of_time": oot["recall"],
})
stability["precision_change_pp"] = (stability["precision_out_of_time"] - stability["precision_in_time"]) * 100
stability["recall_change_pp"] = (stability["recall_out_of_time"] - stability["recall_in_time"]) * 100
show(stability.reset_index(), "In-time versus held-back periods")

print("""
Rank candidates on out-of-time performance; use in-time only to generate them.
A challenger whose advantage shrinks or reverses on periods it has not seen was
fitted to the tuning window.
""")

# %% [markdown]
# ## 8. The decision

# %%
banner("8. METRICS -- PRECISION, RECALL, VOLUME")

metrics_view = comparison[["alerts", "precision", "recall"]].copy()
metrics_view["volume_vs_incumbent"] = (
    comparison["alerts"] / comparison.loc["incumbent: amount > 50k", "alerts"] - 1)
show(metrics_view.reset_index(), "The three numbers the spec asks for, side by side")

print("""
Volume belongs alongside precision and recall, not below them. A challenger
with better precision and worse recall is a judgement call; one that also
triples alert volume is not a judgement call, it is undeliverable, and the
metrics table should make that visible without a separate capacity exercise.
""")

# %% [markdown]
# ## 9. Success criteria: does the challenger merit further validation?

# %%
banner("9. DOES THE CHALLENGER MERIT FURTHER VALIDATION?")

criteria = pd.DataFrame([
    ("Detection", "Recall at equal or lower alert volume", "Hard gate -- no regression without appetite sign-off"),
    ("Typology coverage", "Which typologies the newly-missed cases belong to", "Hard gate -- coverage is assessed per typology"),
    ("Efficiency", "Alerts per true case", "Supporting"),
    ("Out-of-time stability", "Advantage persists on held-back periods", "Hard gate"),
    ("Explainability", "Can an investigator tell why this alerted?", "Hard gate -- affects investigation quality"),
    ("Implementation risk", "Fields available, reliable, in the production engine", "Hard gate"),
    ("Data quality", "Does every input have gaps or defaults?", "Hard gate -- a defaulted field fails closed"),
], columns=["Criterion", "Test", "Weight"])
show(criteria, "What a challenger must clear")

print("""
Note how many are hard gates and how few concern the metric. A challenger
depending on a field that is null for 30% of customers has a real-world recall
far below its backtested recall, because the condition silently fails closed
for those customers -- and fails closed on precisely the population the field
was added to identify. Check field coverage before you check precision.
""")

print(f"""  VERDICT ON THE SPECIFIED CHALLENGER (amount > £30k AND velocity > 5)

  Merits further validation: YES, with conditions.

  FOR
    Precision {comparison.loc['challenger: amount > 30k AND velocity > 5', 'precision']:.2%} against the incumbent's {comparison.loc['incumbent: amount > 50k', 'precision']:.2%}.
    Alert volume falls {1 - comparison.loc['challenger: amount > 30k AND velocity > 5', 'alerts'] / comparison.loc['incumbent: amount > 50k', 'alerts']:.0%}, well inside capacity.
    Effort per case improves from {comparison.loc['incumbent: amount > 50k', 'alerts_per_true_positive']:.1f} to {comparison.loc['challenger: amount > 30k AND velocity > 5', 'alerts_per_true_positive']:.1f} alerts.

  AGAINST
    Recall falls {comparison.loc['incumbent: amount > 50k', 'recall']:.1%} -> {comparison.loc['challenger: amount > 30k AND velocity > 5', 'recall']:.1%}, and the {champion_only:,} newly-missed cases are
    concentrated in high-value low-velocity customers -- a coherent typology,
    not a random sample. Coverage of that typology would have to be re-argued.

  CONDITIONS BEFORE IT PROCEEDS
    1. Field coverage for `velocity` confirmed in the production engine. A null
       velocity fails the AND closed and silently drops the customer.
    2. Parallel run for one quarter, comparing real dispositions on the
       disjoint alerts -- that settles with outcomes what a backtest can only
       estimate from proxy labels.
    3. The variant adding customer risk (row 4) tested alongside it. It recovers
       most of the lost recall at better precision, so validating the specified
       challenger alone would be testing the weaker of two candidates.

  "Merits further validation" is not "adopt". It means the evidence justifies
  spending a quarter and a parallel run on it, which is the decision actually
  in front of you at this stage.
""")

banner("END OF WEEK 6")
print("""
Carry forward into Week 7:
  * Test each risk indicator alone before combining it.
  * AND versus OR decides the operating point more than the thresholds do.
  * Always decompose the overlap, and look at WHO the challenger stops catching.
  * Composite scores buy detection with explainability. Both are real costs.
  * Compare at matched alert volume, and rank on out-of-time performance.
  * Report volume beside precision and recall. Week 8 builds the score properly.
""")
