# 10-Week Advanced Transaction Monitoring Rule Tuning Learning Plan

## Provenance of this plan

The supplied specification covers the learning outcomes, the environment
assumption, and **Weeks 1 to 6**. Those are transcribed here as written.

Weeks 7 to 10 were not in the supplied material. They are derived from the nine
stated learning outcomes — specifically the three that Weeks 1–6 do not cover —
and are marked *(extrapolated)*. If you have the rest of the original
specification, compare it against the mapping below and say where it differs;
the week scripts are independent, so any one can be rewritten without touching
the others.

| Outcome (from the spec) | Covered in | Source |
| --- | --- | --- |
| Design labelled and proxy-labelled backtests | Weeks 1, 2 | spec |
| Tune transaction-monitoring thresholds using evidence | Week 3 | spec |
| Quantify precision/recall trade-offs | Weeks 1, 3 | spec |
| Build alert-volume and risk-yield curves | Weeks 3, 4 | spec |
| Conduct segment-specific calibration | Week 5 | spec |
| Compare incumbent versus challenger rules | Week 6 | spec |
| Identify instability and model drift | Week 7 | *extrapolated* |
| Produce validation-ready tuning papers | Week 9 | *extrapolated* |
| Execute a complete TM backtest and calibration exercise | Week 10 | *extrapolated* |

Week 8 (above/below-the-line testing) is extrapolated and maps to no single
outcome. It is included because the missed-risk estimate it produces is the
evidence a model validator asks for first, and without it Weeks 9 and 10 have
nothing to report on risk acceptance.

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

**Topics**

- Sensitivity analysis
- Threshold sweeps
- Precision-recall trade-offs
- Cost-based optimisation

**Python Exercise**

```python
thresholds = range(10000, 120000, 5000)
```

For each threshold calculate:

- Precision
- Recall
- Alerts generated

Plot results.

**Deliverable**

Produce a recommendation:

- Current threshold = £50k
- Proposed threshold = ?

Supported by evidence.

**Success Criteria.** Document tuning rationale.

> **A note on the fixed grid.** A £5k step is fine for presenting a result and
> poor for finding one: it spends 22 candidates evenly across a range the data
> does not occupy evenly. The week runs the exercise as set, then builds a grid
> from the data's own quantiles and shows what changes.

> Run it: `python weeks/week03_threshold_optimisation.py`

---

## WEEK 4: ALERT VOLUME & CAPACITY MODELLING

**Concept.** The statistically best threshold may be impossible operationally.

**Topics**

- Investigation capacity
- Queue management
- Alert-to-investigator ratios
- Service-level impacts

**Exercise**

Assume:

- 12 investigators
- 25 alerts/day each

```python
capacity = 12 * 25
```

Build alert-volume curves.

**Task.** Determine the highest recall threshold without breaching capacity.

**Success Criteria.** Present a recommendation balancing:

- Risk
- Cost
- Capacity

> Run it: `python weeks/week04_capacity_modelling.py`

---

## WEEK 5: SEGMENT-BASED CALIBRATION

**Concept.** One threshold rarely fits all customer types.

**Topics**

- Retail versus Corporate
- Geography segmentation
- Product segmentation
- Peer groups

**Synthetic Dataset**

```python
df["segment"] = np.where(
    np.random.rand(len(df)) > 0.7,
    "Corporate",
    "Retail"
)
```

**Exercise**

Compare:

- Single threshold
- Segment-specific thresholds

Example:

- Retail = £20k
- Corporate = £100k

**Success Criteria.** Explain improvement achieved without creating unjustified
complexity.

> **A note on this exercise.** `np.random.rand() > 0.7` assigns the segment at
> random, with no reference to any customer attribute, so the two segments have
> near-identical base rates and median amounts. A segmentation uncorrelated with
> behaviour cannot improve detection however its thresholds are set — and at
> matched alert volume a single threshold beats it. The week demonstrates that,
> then repeats the exercise on segments that mean something.

> Run it: `python weeks/week05_segment_calibration.py`

---

## WEEK 6: CHALLENGER RULE DEVELOPMENT

**Concept.** Develop alternative logic to challenge current production settings.

**Topics**

- Incumbent versus challenger
- Risk indicators
- Composite scores
- Explainability

**Scenario**

Current rule:

> Transaction amount > £50k

Challenger:

> Amount > £30k
> **AND**
> Velocity > 5 transactions

**Python Exercise**

Generate:

- `velocity`
- `amount`
- `customer_risk`

Compare both rules.

> Run it: `python weeks/week06_challenger_rules.py`

---

## WEEK 7: STABILITY, INSTABILITY AND MODEL DRIFT *(extrapolated)*

*(The supplied specification ends after Week 6. Weeks 7-10 are extrapolated.)*

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

## WEEK 8: ABOVE AND BELOW-THE-LINE TESTING *(extrapolated)*

**Concept.** Every metric so far measures risk the rule already found.
Below-the-line testing measures the risk it did not.

**Topics**

- Above-the-line (ATL) and below-the-line (BTL) testing
- Sample size planning
- Wilson confidence intervals, and why not the textbook (Wald) interval
- Extrapolating a sample rate to the full below-the-line population

**Success Criteria.** State the missed-risk estimate with a confidence
interval, and size the sample before drawing it.

> Run it: `python weeks/week08_atl_btl_testing.py`

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
