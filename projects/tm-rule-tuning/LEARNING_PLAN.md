# 10-Week Advanced Transaction Monitoring Rule Tuning Learning Plan

## Provenance of this plan

**All ten weeks are transcribed from the supplied specification.** Nothing here
is extrapolated.

Each week records the spec's Concept, Topics, Exercise, Deliverable and Success
Criteria as written. Where a week's runnable script goes beyond the spec, it is
either building the machinery the exercise needs, or flagging something the
exercise itself reveals — noted in a blockquote under that week.

| Outcome (from the spec) | Covered in |
| --- | --- |
| Design labelled and proxy-labelled backtests | Weeks 1, 2 |
| Tune transaction-monitoring thresholds using evidence | Week 3 |
| Quantify precision/recall trade-offs | Weeks 1, 3 |
| Build alert-volume and risk-yield curves | Weeks 3, 4 |
| Conduct segment-specific calibration | Week 5 |
| Compare incumbent versus challenger rules | Week 6 |
| Identify instability and model drift | Week 7 |
| Produce validation-ready tuning papers | Weeks 8, 9 |
| Execute a complete TM backtest and calibration exercise | Week 10 |

### Supplementary material

One topic sits outside the ten weeks, in `extras/`:

**Above and below-the-line (ATL/BTL) testing** — `extras/atl_btl_testing.py`.
Below-the-line sampling is normally the first evidence a model validator asks
for with a threshold change, and it is what turns Week 9's *Limitations — bias
and assumptions* from an assertion into a measured bound. The spec does not
include it, so it is supplementary rather than displacing a week. Run it after
Week 6.

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

## WEEK 7: STABILITY & DRIFT TESTING

**Concept.** A rule that works in one period may fail later.

**Topics**

- Population stability
- Behaviour change
- Data drift
- Seasonality

**Exercise**

Create three periods:

- Year 1
- Year 2
- Year 3

Simulate distribution changes.

Calculate:

- Alert rates
- Precision
- Recall

**Python Technique**

```python
from scipy.stats import ks_2samp
```

**Success Criteria.** Identify unstable thresholds and propose mitigants.

> **A note on the KS test.** Read the KS *statistic*, not the p-value. KS power
> grows with sample size, so on tens of thousands of records the p-value is
> vanishing for any difference at all, including differences far too small to
> move a threshold. The week runs a same-distribution control (a random split of
> Year 1) to establish the noise floor a real comparison has to clear.

> Run it: `python weeks/week07_stability_and_drift.py`

---

## WEEK 8: ADVANCED CALIBRATION USING SCORING

**Concept.** Move beyond a single threshold.

**Topics**

- Risk scoring
- Weighted indicators
- Probability estimates
- Score cut-offs

**Scoring Rule**

```python
score = (
    amount_score * 0.5 +
    velocity_score * 0.3 +
    geo_score * 0.2
)
```

**Exercise**

Create:

- Deciles
- Score distributions
- Yield by decile

**Deliverable.** Determine optimal score cut-off.

**Success Criteria.** Justify chosen cut-off using evidence rather than
intuition.

> **A note on the components.** They must be made commensurable before they are
> weighted. Raw pounds alongside a raw transaction count makes the weights
> decorative — amount decides every alert whatever number sits beside it. The
> week uses percentile ranks for amount and velocity, and an explicit lookup for
> the jurisdiction tier, since a tier is an ordered judgement rather than a
> measurement.

> Run it: `python weeks/week08_scoring_calibration.py`

---

## WEEK 9: VALIDATION & GOVERNANCE

**Concept.** A tuning exercise only succeeds if it is defensible.

**Topics**

- Model governance
- Documentation standards
- Evidence retention
- Independent challenge
- Limitations analysis

**Exercise**

Write a mini tuning paper containing:

**Background**
- Current rule

**Method**
- Data and testing approach

**Results**
- Metrics and analysis

**Recommendation**
- Threshold choice

**Limitations**
- Bias and assumptions

**Success Criteria.** Produce validation-ready documentation.

> Run it: `python weeks/week09_tuning_papers.py`

---

## WEEK 10: CAPSTONE BACKTEST & CALIBRATION PROJECT

**Objective.** Complete an end-to-end transaction-monitoring tuning engagement.

**Capstone Scenario**

Existing rule:

> Cash deposits > £10k in 30 days

Population: 100,000 customers

Fields:

- Customer ID
- Segment
- Country Risk
- Cash Deposits
- Wire Activity
- Velocity
- Alert Outcome
- Case Outcome

Constraints:

- Investigation capacity fixed
- Senior management expects volume reduction
- Missed-risk tolerance limited

**Required Analysis**

**1. Baseline Assessment** — report alert volume, precision, recall, FPR

**2. Threshold Sweep** — evaluate £10k, £15k, £20k, £25k, £30k

**3. Segment Calibration** — compare global, retail and corporate thresholds

**4. Challenger Design** — cash threshold **+** velocity condition

**5. Stability Testing** — run across Q1, Q2, Q3, Q4

**6. Recommendation** — present:

*Proposed Calibration.* Example: Retail = £15k, Corporate = £35k, Velocity > 3

*Impact.*

| Metric | Before | After |
| --- | --- | --- |
| Alerts | | |
| Precision | | |
| Recall | | |
| FPR | | |

*Risks.* Potential blind spots, data limitations, monitoring requirements.

> **A note on the example calibration.** Run faithfully, Retail £15k /
> Corporate £35k / velocity > 3 cuts alert volume by 94% — and uses only 10% of
> the investigation capacity that was supposedly the binding constraint, while
> recall falls 59 percentage points. It satisfies "volume reduction" by
> overshooting it, and fails "missed-risk tolerance limited". The week evaluates
> it as set, then searches the design space for a calibration that actually
> spends the allocation, and recommends that instead.

> **A note on sizing.** Select on the Q4 run-rate, not the twelve-month average.
> The population drifts through the year, so a design averaging inside capacity
> across four quarters can still breach it on the day it goes live — 10 of the
> 85 designs tested do exactly that.

> Run it: `python weeks/week10_capstone.py`

---

## RECOMMENDED WEEKLY STUDY CADENCE

| When | Time | What |
| --- | --- | --- |
| **Monday** | 1 hour | Read theory and regulatory/model-governance materials |
| **Wednesday** | 1 hour | Work through the Python example |
| **Friday** | 1–2 hours | Complete the exercise and document findings |
| **Weekend** | 30 minutes | Write a one-page tuning recommendation |

Roughly 3.5–4.5 hours a week. The Friday and weekend slots are the ones that
build the skill that transfers: running a sweep is quick, but deciding what the
numbers license you to claim — and writing it down so somebody else can
challenge it — is the part the job actually consists of.

---

## STRETCH GOALS (ADVANCED PRACTITIONER LEVEL)

- Bayesian threshold optimisation
- Isolation Forest challengers
- Graph/network-based monitoring
- Dynamic peer-group thresholds
- Population Stability Index (PSI)
- Champion-challenger frameworks
- Explainable machine-learning models for TM optimisation

Two of these are already built into the course rather than left as stretch:
**PSI** runs throughout Week 7 (`tmtuning.stability.psi`), and
**champion-challenger** is Week 6's whole subject. The remaining five are
genuine extensions — the natural next build after Week 10.

---

## RECOMMENDED HABIT

> Maintain a **"Tuning Decision Log"** recording every threshold tested,
> observed metric changes, operational implications, assumptions and final
> rationale. This provides the evidence trail typically expected by model
> validation, compliance governance and audit functions.

This one is implemented rather than merely described: `tmtuning.decision_log`
provides `TuningDecisionLog`, and Week 10 records every option it tests through
it — including the ones it rejects — then exports the log to markdown alongside
the tuning paper.

The reason it matters is narrow and practical. A tuning paper reports the option
that won. The decision log reports the options that lost and why, which is what
somebody needs three years later when the author has moved on and the question
is "why £20,000?". Kept as you go it costs nothing; reconstructed afterwards it
is guesswork dressed as evidence, and a reader can tell which happened.
