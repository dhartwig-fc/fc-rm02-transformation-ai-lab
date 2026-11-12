# 10-Week Advanced Transaction Monitoring Rule Tuning Learning Plan

A practical curriculum taking you from "what is a confusion matrix" to a
complete, validation-ready tuning engagement. Every week is a runnable script:
you read it, run it, and read what it printed.

---

## How to use this plan

```bash
cd projects/tm-rule-tuning
pip install -r requirements.txt
python weeks/week01_tuning_framework.py
```

Each week runs standalone and prints its own working. Charts and generated
documents are written to `outputs/`. The week files are Jupyter percent-format,
so they run as plain Python *and* open as notebooks
(`jupytext --to notebook weeks/*.py`).

Suggested pace is in [Recommended weekly study cadence](#recommended-weekly-study-cadence)
— roughly four hours a week.

**If you are working through the course rather than looking something up, use
[TRAINING_PLAN.md](TRAINING_PLAN.md) instead.** It turns this specification into
a session-by-session schedule, each session built round a worked example with
real numbers. This document stays the reference for what the curriculum covers.

## Provenance

**All ten weeks are transcribed from the supplied specification.** Each week
records the spec's Concept, Topics, Exercise, Deliverable and Success Criteria
as written.

Where a week's script goes beyond the spec it is either building the machinery
the exercise needs, or reporting something the exercise itself reveals. Those
additions are called out in `> blockquotes` under the week concerned, so the
spec and the commentary never blur together.

## The ten weeks

| # | Week | The exercise turns on | Script |
| --- | --- | --- | --- |
| 1 | Building the Tuning Framework | Confusion matrix; why accuracy misleads | [`week01_tuning_framework.py`](weeks/week01_tuning_framework.py) |
| 2 | Rule Backtesting Fundamentals | Alert count, TPs and precision at four thresholds | [`week02_backtesting_fundamentals.py`](weeks/week02_backtesting_fundamentals.py) |
| 3 | Threshold Optimisation | A £10k–£120k sweep and a justified recommendation | [`week03_threshold_optimisation.py`](weeks/week03_threshold_optimisation.py) |
| 4 | Alert Volume & Capacity Modelling | 12 investigators × 25 alerts/day | [`week04_capacity_modelling.py`](weeks/week04_capacity_modelling.py) |
| 5 | Segment-Based Calibration | One threshold versus Retail/Corporate thresholds | [`week05_segment_calibration.py`](weeks/week05_segment_calibration.py) |
| 6 | Challenger Rule Development | Amount > £30k **AND** velocity > 5 | [`week06_challenger_rules.py`](weeks/week06_challenger_rules.py) |
| 7 | Stability & Drift Testing | Three years, `ks_2samp`, seasonality | [`week07_stability_and_drift.py`](weeks/week07_stability_and_drift.py) |
| 8 | Advanced Calibration Using Scoring | A 0.5 / 0.3 / 0.2 weighted score | [`week08_scoring_calibration.py`](weeks/week08_scoring_calibration.py) |
| 9 | Validation & Governance | A five-section mini tuning paper | [`week09_tuning_papers.py`](weeks/week09_tuning_papers.py) |
| 10 | Capstone Backtest & Calibration | 100,000 customers, six-step analysis | [`week10_capstone.py`](weeks/week10_capstone.py) |

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

| Outcome | Covered in |
| --- | --- |
| Design labelled and proxy-labelled backtests | Weeks 1, 2 |
| Tune thresholds using evidence | Week 3 |
| Quantify precision/recall trade-offs | Weeks 1, 3 |
| Build alert-volume and risk-yield curves | Weeks 3, 4 |
| Conduct segment-specific calibration | Week 5 |
| Compare incumbent versus challenger rules | Week 6 |
| Identify instability and model drift | Week 7 |
| Produce validation-ready tuning papers | Weeks 8, 9 |
| Execute a complete backtest and calibration exercise | Week 10 |

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

> **Watch the matrix orientation.** `sklearn` returns `[[TN, FP], [FN, TP]]`, so
> the top-left cell is the *true negative* count, not TP. Published tuning decks
> misread this corner regularly. Unpack with `.ravel()` rather than reading the
> grid by eye.

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

> **Read the result before acting on it.** In the generator above, `case` is
> drawn independently of `amount` — risk and transaction value are statistically
> independent, so precision is flat at the base rate at *every* threshold while
> recall falls. That flat line is the correct answer to the data, and
> recognising it is the week's real skill: a sweep showing no precision lift
> means the variable carries no risk signal, which kills the rule rather than
> inviting you to keep sliding the number. The script demonstrates this, then
> repeats the exercise on a risk-linked population so the constrained
> optimisation has something to find.

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

> **On the fixed grid.** A £5k step is fine for presenting a result and poor for
> finding one: it spends 22 candidates evenly across a range the data does not
> occupy evenly. Roughly half the customers sit below £10k and are never
> examined, while the sparse top of the range is sampled far more finely than
> its handful of customers can support. The week runs the exercise as set, then
> builds a grid from the data's own quantiles and shows what changes.

> **On marginal yield.** Cumulative precision averages in the productive top of
> the distribution; *marginal* precision prices only the alerts you are about to
> add, and is always the worse — and more honest — number. When it falls to the
> population base rate, the next tranche of alerts is no better than random.

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

> **A queue over capacity never settles.** It does not stabilise at a larger
> backlog — it grows without limit for as long as the breach continues. In the
> week's simulation the incumbent rule runs at 4.2× capacity and reaches a
> 61,000 backlog with a 205-day wait inside three months. Its headline 40%
> recall is therefore a paper figure: it counts cases sitting in a queue nobody
> reaches, and a SAR filed on a transaction that old is late.

> **Grid search leaves capacity on the table.** The threshold producing
> *exactly* capacity is a quantile lookup, not a search. Use a grid to
> understand the curve's shape; use the quantile to set the number.

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

> **This segmentation is random.** `np.random.rand() > 0.7` assigns the label
> with no reference to any customer attribute, so the two segments come out with
> near-identical base rates and median amounts. The £20k/£100k split *looks*
> like an improvement — it finds more cases — but only because a £20k retail cut
> is far looser than £50k across 70% of the book. Give the single threshold the
> same alert budget and it wins outright. A segmentation uncorrelated with
> behaviour cannot improve detection however its thresholds are set. The week
> demonstrates that, then repeats on segments that carry real signal.

> **Compare at equal alert budgets, always.** A segmented rule scored against an
> untuned single threshold firing fewer alerts is not a comparison, it is a
> rigged one.

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

**Metrics**

Evaluate:

- Precision
- Recall
- Volume

**Success Criteria.** Determine whether challenger merits further validation.

> **Netted metrics conceal reallocation.** The challenger nearly doubles
> precision and roughly halves recall — but the headline figures net gains
> against losses. Decompose the overlap: the cases it stops catching are
> high-value, low-velocity customers, which is a coherent typology rather than a
> random sample of the incumbent's catch. A challenger that systematically drops
> one typology has narrowed coverage, whatever its precision does.

> **Check field availability before metrics.** A null velocity fails the `AND`
> closed, silently removing that customer from monitoring — and it fails closed
> on precisely the population the field was added to identify. Backtests cannot
> see this, because the test field is always populated.

> **Composite scores appear here in outline only.** Week 8 builds one properly.

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

> **Read the KS statistic, not the p-value.** KS power grows with sample size,
> so on tens of thousands of records the p-value is vanishing for any difference
> at all — including differences far too small to move a threshold. The week
> runs a same-distribution control (a *random* split of Year 1, not a
> first-half/second-half split, which would carry real seasonal signal) to
> establish the noise floor a real comparison must clear.

> **Seasonality and drift look identical in a monitoring pack** and need
> opposite responses. Read the table down a column for the recurring intra-year
> shape, and across a row for year-on-year movement. Compare like calendar
> months, never consecutive ones, before proposing a re-tune.

> **Instability is a property of the threshold's position.** A threshold sitting
> where the distribution is steep converts small behavioural shifts into large
> volume swings; one further out barely moves. Nothing in a single-period
> backtest shows this.

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

> **Make the components commensurable before weighting them.** Raw pounds beside
> a raw transaction count makes the weights decorative — amount would decide
> every alert whatever number sat next to it. The week uses percentile ranks for
> amount and velocity (averaging ties, which matters for a discrete count) and
> an explicit lookup for the jurisdiction tier, since a tier is an ordered
> judgement rather than a measurement.

> **Weights that do not sum to 1 silently rescale the score,** so a fixed
> cut-off drifts in meaning between runs. `tmtuning.scoring.weighted_score`
> refuses them by default.

> **A score has four sets of choices, and a sweep justifies only the last.**
> Component definitions, weights, combination method, then cut-off. Evidence has
> to cover all four — and "0.5 performed best" is circular when the weights were
> chosen on the same data the performance was measured on.

> **Rank boundaries must be fixed from the tuning window.** Percentile ranks
> computed within each period hold alert volume steady by construction, which
> hides exactly the drift Week 7 teaches you to detect.

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

> **A substantial limitations section makes a paper more credible, not less.**
> A validator's job is to find what the author missed; naming the weaknesses
> first turns their findings into confirmations and moves the conversation to
> whether the residual risk is acceptable — which is a decision the risk owner
> can actually take. State the limitation, quantify the *direction* of its bias,
> then say what you did about it.

> **Generate the document from the analysis; never retype it.** Transcription
> errors are the most common defect in tuning papers and are always found by the
> validator rather than the author. It also means anything the analysis did not
> produce cannot quietly appear in the paper.

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

### Required Analysis

**1. Baseline Assessment**

Report:

- Alert volume
- Precision
- Recall
- FPR

**2. Threshold Sweep**

Evaluate:

- £10k
- £15k
- £20k
- £25k
- £30k

**3. Segment Calibration**

Compare:

- Global threshold
- Retail threshold
- Corporate threshold

**4. Challenger Design**

Create:

> Cash threshold
> **+**
> Velocity condition

**5. Stability Testing**

Run across:

- Q1
- Q2
- Q3
- Q4

**6. Recommendation**

Present:

*Proposed Calibration.* Example:

- Retail = £15k
- Corporate = £35k
- Velocity > 3

*Impact.*

| Metric | Before | After |
| --- | --- | --- |
| Alerts | | |
| Precision | | |
| Recall | | |
| FPR | | |

*Risks.*

- Potential blind spots
- Data limitations
- Monitoring requirements

> **The example calibration over-corrects.** Run faithfully, Retail £15k /
> Corporate £35k / velocity > 3 cuts alert volume by 94% — and uses only 10% of
> the investigation capacity that was supposedly the binding constraint, while
> recall falls 59 percentage points and 1,762 additional cases go unalerted. It
> satisfies "volume reduction" by overshooting it and fails "missed-risk
> tolerance limited". The week evaluates it as set, records it in the decision
> log as rejected with reasons, then searches the design space and recommends a
> calibration that spends the allocation — recovering 33pp of recall.

> **Size on the Q4 run-rate, not the twelve-month average.** The population
> drifts through the year, so a design averaging inside capacity across four
> quarters can still breach it on the day it goes live. 10 of the 85 designs
> tested do exactly that.

> **`Alert Outcome` is circular.** It is a function of the incumbent rule: every
> record below £10k reads "NO ALERT" because nobody looked, not because it was
> reviewed and cleared. Treating that as evidence of no risk is the easiest way
> to make a capstone answer look better than it is.

> Run it: `python weeks/week10_capstone.py`

---

## Supplementary material

One topic sits outside the ten weeks, in `extras/`.

**Above and below-the-line (ATL/BTL) testing** —
[`extras/atl_btl_testing.py`](extras/atl_btl_testing.py).

Below-the-line sampling is normally the first evidence a model validator asks
for with a threshold change, and it is what turns Week 9's *Limitations — bias
and assumptions* from an assertion into a measured bound. The spec does not
include it, so it is supplementary rather than displacing a week. Run it after
Week 6.

It also carries two points that transfer directly into the capstone: use Wilson
intervals rather than the textbook Wald interval (which returns a zero-width
`[0, 0]` when a sample turns up no cases, implying certainty that nothing sits
below the line), and never difference two sampled estimates when the effect is
smaller than either one's confidence interval.

---

## Recommended weekly study cadence

| When | Time | What |
| --- | --- | --- |
| **Monday** | 1 hour | Read theory and regulatory/model-governance materials |
| **Wednesday** | 1 hour | Work through the Python example |
| **Friday** | 1–2 hours | Complete the exercise and document findings |
| **Weekend** | 30 minutes | Write a one-page tuning recommendation |

Roughly 3.5–4.5 hours a week.

The Friday and weekend slots build the skill that transfers. Running a sweep is
quick; deciding what the numbers license you to claim, and writing it down so
somebody else can challenge it, is what the job actually consists of.

---

## Stretch goals (advanced practitioner level)

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
genuine extensions, and the natural next build after Week 10.

---

## Recommended habit

> Maintain a **"Tuning Decision Log"** recording every threshold tested,
> observed metric changes, operational implications, assumptions and final
> rationale. This provides the evidence trail typically expected by model
> validation, compliance governance and audit functions.

This one is implemented rather than merely described. `tmtuning.decision_log`
provides `TuningDecisionLog`, and Week 10 records every option it tests through
it — including the ones it rejects — then exports the log to markdown alongside
the tuning paper. A worked example is in
[`docs/example_decision_log.md`](docs/example_decision_log.md).

The reason it matters is narrow and practical. A tuning paper reports the option
that won. The decision log reports the options that lost and why, which is what
somebody needs three years later when the author has moved on and the question
is "why £20,000?".

Kept as you go it costs nothing. Reconstructed afterwards it is guesswork
dressed as evidence — and a reader can always tell which happened.
