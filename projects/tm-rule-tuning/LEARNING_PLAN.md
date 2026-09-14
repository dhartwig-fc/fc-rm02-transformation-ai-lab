# 10-Week Advanced Transaction Monitoring Rule Tuning Learning Plan

## Provenance of this plan

The supplied specification covered the learning outcomes, the environment
assumption, and Weeks 1 to 3 (Week 3 to its Concept line). Those parts are
transcribed here as written.

Weeks 4 to 10 were **not** in the supplied material. They are derived from the
nine stated learning outcomes, mapped to the outcomes that Weeks 1-3 do not
already cover, and are marked *(extrapolated)* throughout. If you have the rest
of the original specification, compare it against the mapping table below and
tell me where it differs — the week scripts are independent of each other, so a
week can be rewritten without touching the others.

| Outcome (from the spec) | Covered in |
| --- | --- |
| Design labelled and proxy-labelled backtests | Weeks 1, 2 |
| Tune transaction-monitoring thresholds using evidence | Week 3 |
| Quantify precision/recall trade-offs | Weeks 1, 4 |
| Build alert-volume and risk-yield curves | Week 4 *(extrapolated)* |
| Conduct segment-specific calibration | Week 5 *(extrapolated)* |
| Identify instability and model drift | Week 7 *(extrapolated)* |
| Compare incumbent versus challenger rules | Week 8 *(extrapolated)* |
| Produce validation-ready tuning papers | Week 9 *(extrapolated)* |
| Execute a complete TM backtest and calibration exercise | Week 10 *(extrapolated)* |

Week 6 (above/below-the-line testing) is extrapolated and maps to no single
outcome. It is included because the missed-risk estimate it produces is the
evidence a model validator asks for first, and without it Weeks 9 and 10 have
nothing to report on risk acceptance.

---

## Learning Outcomes

By the end of 10 weeks you will be able to:

- Design labelled and proxy-labelled backtests
- Tune transaction-monitoring thresholds using evidence
- Quantify precision/recall trade-offs
- Build alert-volume and risk-yield curves
- Conduct segment-specific calibration
- Identify instability and model drift
- Compare incumbent versus challenger rules
- Produce validation-ready tuning papers
- Execute a complete transaction-monitoring backtest and calibration exercise

## Environment Assumption

Python 3.11+, Jupyter Notebook, pandas, numpy, matplotlib, seaborn, scipy,
scikit-learn.

Verify environment:

```python
import pandas as pd
import numpy as np
import sklearn
import matplotlib

print("Environment ready")
```

---

## WEEK 1: BUILDING THE TUNING FRAMEWORK

**Concept.** Before changing thresholds, establish how success will be measured.

**Topics**

- Rule lifecycle
- Backtesting fundamentals
- Alert disposition outcomes
- SAR/STR-based labels versus proxy labels
- Precision, Recall, FPR, F1-score
- Operational capacity constraints

**Worked Example**

Rule: Alert if monthly cash deposits > £10,000

Historical sample:

| Customer | Alert | True Case |
| --- | --- | --- |
| A | Yes | Yes |
| B | Yes | No |
| C | No | Yes |
| D | No | No |

Build a confusion matrix.

**Python Exercise**

```python
from sklearn.metrics import confusion_matrix

y_true = [1, 0, 1, 0]
y_pred = [1, 1, 0, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)
```

**Success Criteria** — Explain:

- False negative risk
- False positive cost
- Why accuracy is a poor metric in AML

> Run it: `python weeks/week01_tuning_framework.py`

---

## WEEK 2: RULE BACKTESTING FUNDAMENTALS

**Concept.** Evaluate existing rule performance using historical data.

**Topics**

- Retrospective testing
- Sample design
- Selection bias
- Coverage analysis
- Alert yield

**Scenario**

Rule: Monthly outbound wires > £50k

Synthetic population:

```python
import numpy as np
import pandas as pd

np.random.seed(42)

n = 5000

df = pd.DataFrame({
    "amount": np.random.gamma(2, 20000, n),
    "case": np.random.binomial(1, 0.04, n),
})
```

**Exercise**

Calculate:

- Alert count
- True positives
- Precision

At thresholds:

- £25k
- £50k
- £75k
- £100k

**Success Criteria.** Identify threshold that maximises risk detection under a
fixed alert volume.

> **A note on this exercise.** In the generator above, `case` is drawn
> independently of `amount`. Risk and transaction value are statistically
> independent, so precision is flat at the base rate at *every* threshold while
> recall falls. That flat line is the correct answer to the data, and
> recognising it is the skill the week builds. Week 2's script demonstrates
> this, then repeats the exercise on a risk-linked population so the
> constrained optimisation has something to find.

> Run it: `python weeks/week02_backtesting_fundamentals.py`

---

## WEEK 3: THRESHOLD OPTIMISATION

**Concept.** Tune thresholds systematically rather than relying on expert
judgement alone.

*(The supplied specification ends here. What follows is extrapolated.)*

**Topics** *(extrapolated)*

- Systematic threshold sweeps versus judgement-led adjustment
- Quantile grids and why linear grids mislead on skewed data
- Marginal yield — pricing the alerts you are about to add
- Constrained optimisation against operational capacity
- Overfitting and out-of-time validation

**Success Criteria.** Recommend a threshold, state the constraint it was chosen
under, and show that it holds on data it was not tuned on.

> Run it: `python weeks/week03_threshold_optimisation.py`

---

## WEEK 4: ALERT VOLUME AND RISK YIELD CURVES *(extrapolated)*

**Concept.** Turn the sweep into the two curves a decision is actually made
from, and find the point where the trade stops being worth it.

**Topics**

- Alert volume curves and the "knee"
- Volume elasticity, and why a steep threshold is a fragile one
- Risk yield curves; the precision-recall frontier
- Costing a threshold: investigator effort versus missed risk
- Sensitivity of any "optimum" to its least evidenced input

**Success Criteria.** Produce the three charts a tuning paper needs, and defend
a recommended operating point using them.

> Run it: `python weeks/week04_volume_and_yield_curves.py`

---

## WEEK 5: SEGMENT-SPECIFIC CALIBRATION *(extrapolated)*

**Concept.** One threshold across a heterogeneous portfolio is a compromise that
serves no segment well. Quantify the cost and allocate a fixed alert budget
where it detects the most.

**Topics**

- Segment profiling — when segmentation is and is not justified
- Per-segment yield curves on per-segment grids
- Allocating a fixed alert budget across segments
- The governance cost of more thresholds

**Success Criteria.** Show the detection uplift from segment calibration at an
unchanged total alert volume — or show that there isn't one.

> Run it: `python weeks/week05_segment_calibration.py`

---

## WEEK 6: ABOVE AND BELOW-THE-LINE TESTING *(extrapolated)*

**Concept.** Every metric so far measures risk the rule already found.
Below-the-line testing measures the risk it did not.

**Topics**

- Above-the-line (ATL) and below-the-line (BTL) testing
- Sample size planning
- Wilson confidence intervals, and why not the textbook (Wald) interval
- Extrapolating a sample rate to the full below-the-line population

**Success Criteria.** State the missed-risk estimate with a confidence
interval, and size the sample before drawing it.

> Run it: `python weeks/week06_atl_btl_testing.py`

---

## WEEK 7: STABILITY, INSTABILITY AND MODEL DRIFT *(extrapolated)*

**Concept.** A threshold is tuned against one snapshot of behaviour. Behaviour
then moves.

**Topics**

- Period-by-period performance versus a pooled figure
- Population Stability Index (PSI)
- Control charts, and where the limits must come from
- Distinguishing population drift from rule decay
- Setting monitoring triggers that can actually fire

**Success Criteria.** Detect drift in a population you were not told was
drifting, and say which kind it is.

> Run it: `python weeks/week07_stability_and_drift.py`

---

## WEEK 8: INCUMBENT VERSUS CHALLENGER RULES *(extrapolated)*

**Concept.** Comparing a proposed rule to the one in production is not a metrics
table. Netted metrics conceal reallocation, and reallocation is where the risk
decision lives.

**Topics**

- Like-for-like comparison design
- Alert overlap — what the challenger uniquely finds and uniquely misses
- Composite and multi-condition challengers
- Out-of-time confirmation
- Field availability as a hard gate

**Success Criteria.** Recommend for or against a challenger, and be able to say
exactly what risk the change accepts.

> Run it: `python weeks/week08_champion_challenger.py`

---

## WEEK 9: PRODUCING VALIDATION-READY TUNING PAPERS *(extrapolated)*

**Concept.** The analysis is not the deliverable. A tuning change is approved or
rejected on a document.

**Topics**

- The ten areas independent validation checks
- The standard section structure
- Writing limitations that strengthen rather than weaken a paper
- Generating the paper from the analysis rather than retyping it

**Success Criteria.** Produce a complete tuning paper, generated directly from
analysis outputs, that answers a validator's questions before they ask.

> Run it: `python weeks/week09_tuning_papers.py`

---

## WEEK 10: CAPSTONE — COMPLETE BACKTEST AND CALIBRATION EXERCISE *(extrapolated)*

**The brief.**

> Rule TM-014 (*monthly outbound wires > £50,000*) has been in production for
> two years without re-tuning. Operations report the alert queue is growing and
> investigators say yield has fallen. Capacity is 350 alerts per month and there
> is no budget for more.
>
> Determine whether the rule is still fit for purpose, recommend a calibration,
> and produce a paper for independent validation.

**Steps**

1. Build the sample and profile it
2. Backtest the incumbent; diagnose drift versus decay
3. Sweep and select under the capacity constraint
4. Validate out of time
5. Calibrate by segment
6. Test below the line — and measure the incremental change directly
7. Compare against a challenger
8. Recommend, and generate the paper

**Success Criteria.** A defensible recommendation, evidence for each of the ten
validation areas from Week 9, and a generated paper.

> Run it: `python weeks/week10_capstone.py`
