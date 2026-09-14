# %% [markdown]
# # Week 8 -- Incumbent versus Challenger Rules
#
# **Concept:** Comparing a proposed rule to the one in production is not a
# metrics table. Netted metrics conceal reallocation, and reallocation is where
# the risk decision lives.
#
# **Topics**
#
# * Like-for-like comparison design
# * Alert overlap -- what the challenger uniquely finds, and uniquely misses
# * Composite and multi-condition challengers
# * Out-of-time confirmation
# * Deciding on evidence rather than on the headline number
#
# **Success criteria.** Recommend for or against a challenger, and be able to
# say exactly what risk the change accepts.

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

from tmtuning import (answer, banner, apply_rule, compare_rules, generate_population,
                      rule_overlap, show, split_by_period)

CHAMPION_THRESHOLD = 50_000

# %%
banner("1. THE CANDIDATES")

population = generate_population(n=40_000, seed=808, n_periods=12)
in_time, out_of_time = split_by_period(population, holdout_periods=3)


def build_rules(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Define every candidate once, so the same logic is scored on both samples."""
    wires = df["monthly_wire_value"]
    return {
        # Incumbent: a single value threshold.
        "champion": apply_rule(wires, CHAMPION_THRESHOLD),
        # A: same shape, tighter. The usual "reduce volume" proposal.
        "challenger_A_tighter": apply_rule(wires, 75_000),
        # B: value OR risk-indicator. Adds a second detection angle.
        "challenger_B_risk_overlay": (
            apply_rule(wires, 75_000)
            | ((wires > 25_000) & (df["high_risk_jurisdiction"] == 1))
            | ((wires > 25_000) & (df["pep_flag"] == 1))
        ),
        # C: value AND a behavioural qualifier. Narrows on cash intensity.
        "challenger_C_cash_qualified": apply_rule(wires, 40_000) & (df["cash_ratio"] > 0.25),
    }


rules = build_rules(in_time)
for name, flags in rules.items():
    print(f"  {name:<30} {int(np.asarray(flags).sum()):>6,} alerts")

print("""
All four are scored on the SAME sample with the SAME labels. Running the
champion on last year's data and the challenger on this year's is the most
common way a challenger is made to look good, and it is usually accidental.
""")

# %% [markdown]
# ## 2. The headline comparison

# %%
banner("2. HEADLINE METRICS")

comparison = compare_rules(in_time, rules, champion="champion")
show(comparison[["alerts", "tp", "fn", "precision", "recall",
                 "alerts_per_true_positive", "delta_tp", "delta_alerts"]].reset_index(),
     "In-time performance, deltas against the champion")

# %% [markdown]
# ## 3. Break the netting open

# %%
banner("3. ALERT OVERLAP -- WHAT NETTING HIDES")

for name in ["challenger_A_tighter", "challenger_B_risk_overlay", "challenger_C_cash_qualified"]:
    overlap = rule_overlap(in_time, rules["champion"], rules[name])
    show(overlap, f"\nChampion vs {name}")
    champion_only = overlap.loc[overlap["cell"] == "Champion only", "cases"].iloc[0]
    challenger_only = overlap.loc[overlap["cell"] == "Challenger only", "cases"].iloc[0]
    net = challenger_only - champion_only
    print(f"  -> newly missed {champion_only:,} cases, newly caught {challenger_only:,}, "
          f"net {net:+,}")

print("""
The net figure is the one that appears in a summary slide. The two gross
figures are the ones that matter. A challenger that nets +5 cases by losing
40 and gaining 45 is not an improvement to an existing rule -- it is a
different rule detecting different behaviour, and it needs typology review
before anyone looks at its precision.
""")

answer("A challenger nets +3 cases but newly misses 40. Do you recommend it?",
       """
Not on those numbers alone, and the reason is that the 40 are not
interchangeable with the 45 it gains.

First ask what the 40 have in common. If they cluster on a typology the
rule exists to cover -- the one named in its design rationale -- then the
challenger has quietly narrowed the bank's coverage of that typology, and
a +3 net is not compensation for it. Detection coverage is assessed per
typology, not in aggregate.

Second, 3 out of a few hundred is inside sampling noise. Re-run on the
out-of-time sample: if the sign flips, there was never a difference to
recommend.

The usual right answer is to run both in parallel for a period and compare
dispositions on the disjoint alerts, which costs a quarter and settles the
question with investigation outcomes rather than proxy labels.
""")

# %% [markdown]
# ## 4. Does it hold out of time?

# %%
banner("4. OUT-OF-TIME CONFIRMATION")

oot_rules = build_rules(out_of_time)
oot = compare_rules(out_of_time, oot_rules, champion="champion")

# Rates, not counts: the holdout spans fewer periods, so its raw counts are
# lower for every rule by construction and comparing them says nothing.
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
A challenger whose advantage shrinks or reverses out of time was fitted to
the tuning window. Rank candidates on out-of-time performance; use in-time
only to generate them.
""")

# %% [markdown]
# ## 5. The decision

# %%
banner("5. DECISION FRAMEWORK")

criteria = pd.DataFrame([
    ("Detection", "Recall at equal or lower alert volume", "Hard gate -- no regression without explicit appetite sign-off"),
    ("Typology coverage", "Which typologies the newly-missed cases belong to", "Hard gate -- coverage is assessed per typology"),
    ("Efficiency", "Alerts per true case", "Supporting"),
    ("Out-of-time stability", "Advantage persists on held-back periods", "Hard gate"),
    ("Explainability", "Can an investigator tell why this alerted?", "Hard gate -- affects investigation quality"),
    ("Implementation risk", "Fields available, reliable, and in the production engine", "Hard gate"),
    ("Data quality", "Does the new field have gaps or defaults?", "Hard gate -- a defaulted field silently disables the condition"),
], columns=["Criterion", "Test", "Weight"])
show(criteria, "What a challenger must clear")

print("""
Note how many are hard gates and how few are about the metric. A challenger
depending on a field that is null for 30% of customers has a real-world
recall far below its backtested recall, because the condition silently fails
closed for those customers. Check field coverage before you check precision.
""")

banner("END OF WEEK 8")
print("""
Carry forward into Week 9:
  * Same sample, same labels, same period, or it is not a comparison.
  * Always decompose the overlap. Net figures hide reallocation.
  * Rank on out-of-time performance.
  * Check field availability before metrics -- a null field fails closed.
""")
