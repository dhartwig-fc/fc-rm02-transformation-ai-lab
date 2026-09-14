# %% [markdown]
# # Week 9 -- Producing Validation-Ready Tuning Papers
#
# **Concept:** The analysis is not the deliverable. A tuning change is approved
# or rejected on a document, and analysis that cannot be written up in the
# expected shape does not get implemented.
#
# **Topics**
#
# * What independent validation actually checks
# * The standard section structure
# * Writing limitations that strengthen rather than weaken a paper
# * Generating the paper from the analysis, not retyping it
#
# **Success criteria.** Produce a complete tuning paper, generated directly
# from analysis outputs, that answers a validator's questions before they ask.

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

from tmtuning import (answer, banner, apply_rule, btl_test, compare_rules,
                      compare_uniform_vs_segmented, generate_population, optimise_threshold,
                      rule_overlap, save_paper, show, stability_report, threshold_sweep,
                      tuning_paper)

OUT = _ROOT / "outputs"
CAPACITY = 2_500
CURRENT_THRESHOLD = 50_000

# %% [markdown]
# ## 1. What validation actually checks

# %%
banner("1. WHAT INDEPENDENT VALIDATION CHECKS")

checks = pd.DataFrame([
    ("Conceptual soundness", "Does the rule detect the typology it claims to?",
     "Design rationale traced to a typology, not just to a number"),
    ("Data quality", "Is the tuning sample complete and accurate?",
     "Record counts reconciled to source; field coverage stated"),
    ("Label integrity", "What counts as a true case, and how sparse is it?",
     "Explicit definition, observation window, and known biases"),
    ("Sample design", "Is the sample representative and free of selection bias?",
     "Period span, inclusion rules, treatment of exited customers"),
    ("Methodology", "Is the threshold chosen by a defensible procedure?",
     "Stated objective AND constraint; the sweep itself"),
    ("Outcome analysis", "What does the change do to detection?",
     "Volume, precision, recall, and missed-risk estimate"),
    ("Below-the-line", "What risk does this accept?",
     "Sampled BTL test with a confidence interval"),
    ("Stability", "Does it hold over time and out of sample?",
     "Period-by-period results; out-of-time validation"),
    ("Ongoing monitoring", "How will decay be detected?",
     "Named metrics, trigger levels, frequencies and actions"),
    ("Limitations", "What does the author know is weak?",
     "Stated explicitly -- see section 3"),
], columns=["Area", "The question", "What satisfies it"])
show(checks, "Ten areas, and what closes each one")

print("""
Note that only two of the ten are about the threshold number. Most of a
validator's time goes on whether the evidence supports ANY conclusion, and
the most common reason a paper is returned is not a wrong threshold -- it is
an unstated label definition or an unquantified below-the-line position.
""")

# %% [markdown]
# ## 2. Run the analysis

# %%
banner("2. ANALYSIS")

population = generate_population(n=40_000, seed=909, n_periods=12)
sweep = threshold_sweep(population, "monthly_wire_value", "case", n_thresholds=40)

proposed = optimise_threshold(sweep, objective="recall", max_alerts=CAPACITY)
# Score the live threshold exactly, rather than snapping it to the nearest
# grid point -- a paper that says "move from £50,482" when the rule in
# production reads £50,000 invites a correction before anyone reaches the
# recommendation.
current = threshold_sweep(population, "monthly_wire_value", "case",
                         thresholds=[CURRENT_THRESHOLD]).iloc[0]

print(f"  Current threshold   £{current['threshold']:,.0f} "
      f"({int(current['alerts']):,} alerts, {current['recall']:.1%} recall)")
print(f"  Proposed threshold  £{proposed['threshold']:,.0f} "
      f"({int(proposed['alerts']):,} alerts, {proposed['recall']:.1%} recall)")

proposed_flags = apply_rule(population["monthly_wire_value"], float(proposed["threshold"]))
btl = btl_test(population, proposed_flags, n_below=1_500, seed=909)
segments = compare_uniform_vs_segmented(population, capacity=CAPACITY)
stability = stability_report(population, float(proposed["threshold"]), baseline_periods=3)

challenger_flags = {
    "champion (current)": apply_rule(population["monthly_wire_value"], float(current["threshold"])),
    "proposed": proposed_flags,
}
challengers = compare_rules(population, challenger_flags, champion="champion (current)")
overlap = rule_overlap(population, challenger_flags["champion (current)"], proposed_flags)

print(f"  BTL missed-risk estimate: {btl['estimated_missed_cases']:,.0f} "
      f"(upper bound {btl['estimated_missed_cases_upper']:,.0f})")
print(f"  Stability breaches: {int(stability['any_breach'].sum())} of {len(stability)} periods")

# %% [markdown]
# ## 3. Limitations that strengthen a paper

# %%
banner("3. WRITING LIMITATIONS")

print("""
Counter-intuitive but reliable: a paper with a substantial limitations
section is treated as MORE credible, not less.

A validator's job is to find what the author missed. If the paper already
names the weaknesses, the validator's findings become confirmations rather
than discoveries, and the conversation moves to whether the residual risk is
acceptable -- which is a decision the risk owner can actually take.

A paper with no limitations section says one of two things: the author did
not look, or the author looked and chose not to say. Both get the paper
returned, and the second damages the author's next one too.

Limitations to state by default, because they are true of essentially every
transaction monitoring tuning exercise:
""")

limitations = pd.DataFrame([
    ("Label sparsity", "SARs are rare and lag the alert period",
     "Recall is measured against a partial label set and is optimistic"),
    ("Label circularity", "SARs exist only where someone investigated",
     "Below-the-line labels are systematically missing"),
    ("Backtest assumption", "History is assumed representative of the forward period",
     "Any known upcoming change invalidates this"),
    ("Single-rule view", "The rule is tuned in isolation",
     "Coverage overlap with other rules is not assessed here"),
    ("Proxy for harm", "A case is not a measure of the value laundered",
     "Detection counts do not weight by severity"),
], columns=["Limitation", "Why it exists", "What it means for the numbers"])
show(limitations, "Standing limitations")

answer("How do you write a limitation without undermining the recommendation?",
       """
State the limitation, quantify its direction, then say what you did about it.

Weak:    "Labels may be incomplete."
Strong:  "SAR labels are absent below the line by construction, which biases
          measured recall upward. A 1,500-record below-the-line sample bounds
          the missed-case rate at 2.4% with 95% confidence, and the
          recommendation is made against that upper bound rather than the
          point estimate."

The second names the same weakness and closes it. The direction of the bias
matters most: a reviewer can accept an estimate they know is optimistic if
they are told by how much and in which direction.
""")

# %% [markdown]
# ## 4. Generate the paper

# %%
banner("4. GENERATING THE PAPER")

paper = tuning_paper(
    rule_name="TM-014 High Value Outbound Wires",
    rule_logic="ALERT IF monthly_outbound_wire_value > threshold\n"
               "  scoped to: all active customers\n"
               "  frequency: monthly, run on the 1st for the preceding calendar month",
    population=population,
    sweep=sweep,
    proposed=proposed,
    current=current,
    btl=btl,
    segment_comparison=segments,
    stability=stability,
    challenger=challengers,
    overlap=overlap,
    author="Dan Hartwig",
    label_basis="Confirmed SAR/STR submission within 90 days of the alert period",
    limitations=[
        "Coverage overlap with TM-009 (rapid movement of funds) has not been "
        "assessed; some cases counted here may also be detected by that rule.",
    ],
)

path = save_paper(OUT / "week09_tuning_paper.md", paper)
print(f"Written to {path}  ({len(paper.splitlines())} lines)")
print("\n--- first 30 lines ---")
print("\n".join(paper.splitlines()[:30]))

# %%
banner("5. THE POINT OF GENERATING IT")

print("""
The paper was generated from the analysis objects, not retyped from them.
That matters for three reasons:

  1. Transcription errors are the most common defect in tuning papers, and
     they are always found by the validator rather than the author.
  2. Re-running the analysis on corrected data regenerates the document,
     so the paper cannot silently fall out of date with its own evidence.
  3. Anything the analysis did not produce cannot appear in the paper --
     which makes an omission visible instead of quietly fillable by hand.

Note how the generator handles missing evidence: pass no BTL result and the
section does not appear, but a line appears in the limitations section saying
missed risk is unquantified. Skipping work should cost you a sentence in the
document, not go unnoticed.
""")

no_btl = tuning_paper(
    rule_name="TM-014 (illustration: no BTL test run)",
    rule_logic="ALERT IF monthly_outbound_wire_value > threshold",
    population=population, sweep=sweep, proposed=proposed, current=current,
)
print("Limitations recorded when evidence is missing:\n")
in_limitations = False
for line in no_btl.splitlines():
    if line.startswith("## ") and "Limitations" in line:
        in_limitations = True
        continue
    if in_limitations:
        if line.startswith("## "):
            break
        if line.strip().startswith("-"):
            print(f"  {line.strip()}")

banner("END OF WEEK 9")
print("""
Carry forward into Week 10:
  * Only two of ten validation areas concern the threshold itself.
  * Quantify the direction of every bias you disclose.
  * Generate the document from the analysis. Never retype it.
  * Week 10 runs the whole method end to end.
""")
