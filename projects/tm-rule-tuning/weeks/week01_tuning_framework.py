# %% [markdown]
# # Week 1 -- Building the Tuning Framework
#
# **Concept:** Before changing thresholds, establish how success will be measured.
#
# **Topics**
#
# * Rule lifecycle
# * Backtesting fundamentals
# * Alert disposition outcomes
# * SAR/STR-based labels versus proxy labels
# * Precision, Recall, FPR, F1-score
# * Operational capacity constraints
#
# **Success criteria.** By the end of this week you can explain false negative
# risk, false positive cost, and why accuracy is a poor metric in AML.
#
# Run as a script (`python weeks/week01_tuning_framework.py`) or open as a
# notebook -- the `# %%` markers make it a valid Jupyter percent-format file.

# %%
# --- path bootstrap: lets this file run without installing the package ---
import pathlib
import sys

for _p in pathlib.Path(__file__ if "__file__" in globals() else "x").resolve().parents:
    if (_p / "src" / "tmtuning").is_dir():
        sys.path.insert(0, str(_p / "src"))
        break

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from tmtuning import answer, banner, classification_metrics, confusion_frame, show

# %% [markdown]
# ## 1. The rule lifecycle
#
# Tuning is one stage of a loop, not a one-off task. Knowing which stage you are
# in tells you what evidence you owe.

# %%
banner("1. RULE LIFECYCLE")

lifecycle = pd.DataFrame([
    ("1. Design", "Typology and risk the rule is meant to detect", "Typology note, risk appetite"),
    ("2. Build", "Rule logic and parameters implemented", "Rule spec, data lineage"),
    ("3. Deploy", "Rule live, generating alerts", "Implementation evidence"),
    ("4. Operate", "Alerts triaged, investigated, dispositioned", "Disposition data"),
    ("5. Monitor", "Volume and yield tracked against expectation", "Monitoring pack"),
    ("6. Tune", "Thresholds re-evaluated against outcomes", "Tuning paper"),
    ("7. Validate", "Independent challenge of the tuning", "Validation report"),
], columns=["Stage", "What happens", "Evidence produced"])
show(lifecycle, "The loop -- tuning feeds validation, which feeds the next design cycle")

print("\nTuning without stage 4 data is guesswork. If alert dispositions are not")
print("captured reliably, fix that before touching a threshold -- an unlabelled")
print("backtest cannot tell you whether a change helped or harmed.")

# %% [markdown]
# ## 2. Labels: what counts as a "true case"?
#
# Every metric you compute is conditional on this choice. Get it explicit and
# written down before you compute anything.

# %%
banner("2. LABEL DEFINITIONS")

labels = pd.DataFrame([
    ("SAR/STR filed", "Strong", "Sparse, slow, and itself a product of what got alerted",
     "The defensible default. State the observation window."),
    ("Escalated to L2/L3", "Medium", "Measures investigator behaviour, not confirmed risk",
     "Useful proxy when SAR volume is too low to tune on."),
    ("Law enforcement request / production order", "Strong", "Very sparse; arrives late",
     "Good corroboration, too thin to tune on alone."),
    ("Account exited for financial crime reasons", "Medium-strong", "Conflates risk with commercial decisions",
     "Cross-check against exit reason codes."),
    ("Closed no action", "Negative label", "Absence of proof, not proof of absence",
     "Do not treat as confirmed clean."),
], columns=["Label source", "Strength", "Weakness", "How to use it"])
show(labels, "SAR/STR-based labels versus proxy labels")

print("\nThe circularity to name out loud in any tuning paper:")
print("a SAR can only be filed on a case someone looked at, and someone only")
print("looks at what got alerted. So SAR labels are systematically missing")
print("below the line. That biases measured recall UPWARDS -- the rule looks")
print("better at catching risk than it is. Week 8's below-the-line testing")
print("exists specifically to put a bound on that bias.")

# %% [markdown]
# ## 3. Worked example
#
# > **Rule:** Alert if monthly cash deposits > £10,000
#
# | Customer | Alert | True Case |
# | --- | --- | --- |
# | A | Yes | Yes |
# | B | Yes | No |
# | C | No | Yes |
# | D | No | No |
#
# Build a confusion matrix.

# %%
banner("3. WORKED EXAMPLE -- CONFUSION MATRIX")

sample = pd.DataFrame({
    "customer": ["A", "B", "C", "D"],
    "alert": [1, 1, 0, 0],
    "true_case": [1, 0, 1, 0],
})
show(sample, "Historical sample")

print("\nOne customer in each cell:")
print("  A -> True positive   : alerted, and was a case          (the rule worked)")
print("  B -> False positive  : alerted, was not a case          (wasted investigation)")
print("  C -> False negative  : not alerted, but was a case      (MISSED RISK)")
print("  D -> True negative   : not alerted, was not a case      (correctly left alone)")

show(confusion_frame(sample["true_case"], sample["alert"]),
     "\nConfusion matrix, laid out as a reviewer expects", index=True)

# %% [markdown]
# ## 4. Python exercise (as set in the spec)

# %%
banner("4. PYTHON EXERCISE")

y_true = [1, 0, 1, 0]
y_pred = [1, 1, 0, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\nRead that output carefully -- this trips people up constantly.")
print("sklearn orders the matrix [[TN, FP], [FN, TP]] with labels ascending,")
print("so the value in the TOP-LEFT is the TRUE NEGATIVE count, not TP.")
tn, fp, fn, tp = cm.ravel()
print(f"  TN={tn}   FP={fp}   FN={fn}   TP={tp}")
print("\nMany published tuning decks misread this corner and report their")
print("true positives and true negatives swapped. Always unpack with .ravel()")
print("or pass labels= explicitly rather than reading the grid by eye.")

# %% [markdown]
# ## 5. The metric set

# %%
banner("5. METRICS")

metrics = classification_metrics(y_true, y_pred)
for name in ["precision", "recall", "fpr", "f1", "specificity", "accuracy"]:
    print(f"  {name:<12} {metrics[name]:.4f}")

print("""
In tuning language:
  precision = ALERT YIELD      -- of what we raise, how much is productive
  recall    = DETECTION RATE   -- of the risk present, how much we catch
  FPR       = noise rate against the clean population
  F1        = a tie-breaker, never a target (see below)
""")

# %% [markdown]
# ## 6. Success criteria
#
# Explain false negative risk, false positive cost, and why accuracy is a poor
# metric in AML.

# %%
banner("6. SUCCESS CRITERIA")

answer("What is false negative risk?", """
A false negative is laundered money that moved through the bank without
anyone looking. The cost is not a missed statistic -- it is unreported
criminal proceeds, a regulatory finding on monitoring coverage, potential
enforcement, and the harm that sits behind the transaction.
It is also the risk you cannot see in production. False positives announce
themselves in the alert queue; false negatives are silent by construction.
That asymmetry is why tuning must actively test below the line (Week 8),
rather than trusting that a quiet metric means a clean portfolio.
""")

answer("What is false positive cost?", """
Every unproductive alert consumes investigator hours that could have been
spent on real risk. The cost compounds: a backlog builds, average
investigation quality falls under time pressure, genuine cases sit unworked
for longer, and investigators start to distrust the rule -- which quietly
degrades the quality of review on the alerts that ARE productive.
So false positives are not merely an efficiency problem. Past a certain
volume they become a detection problem too.
""")

answer("Why is accuracy a poor metric in AML?", """
Because the base rate is tiny. At a 3% case rate, a rule that alerts on
nothing is 97% accurate while detecting zero risk. Accuracy is dominated
by the true negative cell, which is the one cell nobody is paying for.
Demonstration below.
""")

# %%
n = 10_000
prevalence = 0.03
rng = np.random.default_rng(42)
truth = rng.binomial(1, prevalence, n)

alert_nothing = np.zeros(n, dtype=int)
alert_everything = np.ones(n, dtype=int)

comparison = pd.DataFrame([
    {"rule": "Alert on nothing", **classification_metrics(truth, alert_nothing)},
    {"rule": "Alert on everything", **classification_metrics(truth, alert_everything)},
])
show(comparison[["rule", "alerts", "tp", "fn", "accuracy", "precision", "recall"]],
     f"Two useless rules on a {prevalence:.0%} base rate population")

print("\n'Alert on nothing' scores 97% accuracy and misses 100% of the risk.")
print("It would pass an accuracy-based test and fail the only test that matters.")
print("Never put accuracy in a tuning paper. Use precision, recall, alert volume")
print("and effort per true case.")

# %% [markdown]
# ## 7. Operational capacity -- the constraint that makes this real

# %%
banner("7. OPERATIONAL CAPACITY")

investigators = 12
alerts_per_investigator_per_day = 8
working_days = 21
monthly_capacity = investigators * alerts_per_investigator_per_day * working_days

print(f"  {investigators} investigators x {alerts_per_investigator_per_day} alerts/day "
      f"x {working_days} days = {monthly_capacity:,} alerts per month")
print(f"""
This single number reframes the whole exercise. The question is never
"what is the best threshold?" in the abstract. It is:

    maximise detection, subject to alerts <= {monthly_capacity:,} per month

A threshold that detects more risk but generates {monthly_capacity * 2:,} alerts does not
detect more risk -- it generates a backlog, and the extra alerts age
unworked. Capacity is the constraint every later week optimises against.
""")

banner("END OF WEEK 1")
print("""
Carry forward into Week 2:
  * Metrics are conditional on a label definition. Write yours down.
  * SAR labels are missing below the line; recall is optimistic by default.
  * Accuracy is banned. Precision, recall, volume, effort per case.
  * Tuning is constrained optimisation, and capacity is the constraint.
""")
