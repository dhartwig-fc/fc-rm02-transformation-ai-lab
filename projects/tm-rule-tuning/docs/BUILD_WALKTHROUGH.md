# What Was Built — A Walkthrough

Training material for the codebase itself: what each piece does, why it exists,
and the real output that proves it works.

Read this once before Week 1. It is not the course — it is the tour of the
workshop you are about to work in.

---

## Where this sits

| Document | Answers |
| --- | --- |
| `LEARNING_PLAN.md` | *What does the curriculum cover?* — your specification, transcribed |
| `TRAINING_PLAN.md` | *What do I do on Monday?* — the session-by-session schedule |
| **`docs/BUILD_WALKTHROUGH.md`** | ***What did you build, and how does it work?*** — this file |

---

## 1. The tools

### What is declared

```
pandas>=2.2,<4        numpy>=1.26,<3        matplotlib>=3.8,<4
seaborn>=0.13,<1      scipy>=1.11,<2        scikit-learn>=1.4,<2
pytest>=8.0
```

### What is actually used

| Library | Where it is used | How much |
| --- | --- | --- |
| **pandas** | Everywhere — every table, every sweep, every report | Core |
| **numpy** | Everywhere — all the arithmetic and the random generators | Core |
| **matplotlib** | `plots.py` only | Core, one module |
| **scikit-learn** | `weeks/week01` — one call to `confusion_matrix` | Once |
| **scipy** | `weeks/week07` — one call to `ks_2samp` | Once |
| **seaborn** | **Nowhere.** Zero imports across the whole project | Never |
| **pytest** | `tests/` — 143 tests | Core |

Three things worth understanding from that table.

**The toolkit is deliberately thin.** `src/tmtuning/` — all 2,814 lines of it —
imports only pandas, numpy and matplotlib. Every metric, sweep, allocation,
stability test and report is written out rather than pulled from a library. That
is a teaching choice: you can read the arithmetic, and a validator can too. A
tuning result you cannot re-derive by hand is a tuning result you cannot defend.

**sklearn and scipy appear exactly where your spec names them.** Week 1's
`confusion_matrix` and Week 7's `ks_2samp` are in the specification as written,
so the weeks use them. Nothing else does. `challenger.py` even hand-rolls the
inverse error function (`_erfinv`) rather than importing `scipy.stats.norm`,
with a comment saying so, to keep the library's dependency surface at three.

**seaborn is declared and never imported.** Your spec's Environment Assumption
lists it, so it is in `requirements.txt` for fidelity with the spec. But nothing
uses it — matplotlib does all the charting. It is a harmless but real
discrepancy: either drop it from requirements, or use it somewhere. Flagged
rather than quietly left.

---

## 2. How a week works

Every week script follows the same shape. Nothing is hidden in a framework.

```
weeks/weekNN_topic.py                 the lesson — runs top to bottom, prints its working
        │
        │  imports
        ▼
src/tmtuning/                         the toolkit — reusable, tested, documented
   metrics · data · thresholds
   segments · challenger · stability
   scoring · plots · reporting · decision_log
        │
        │  writes
        ▼
outputs/                              charts (.png), papers (.md), decision logs
```

A week script is a **percent-format** file: the `# %%` markers mean it runs as
plain Python *and* opens as a Jupyter notebook
(`jupytext --to notebook weeks/*.py`).

Each begins with a six-line path bootstrap so it runs without installing
anything — it walks up the directory tree looking for `src/tmtuning`. That is
why `python weeks/week01_tuning_framework.py` works from a fresh clone.

---

## 3. The library, module by module

Each module below: the problem it solves, the one call you will actually use,
and the decision it encodes that you would otherwise have to make yourself.

### `metrics.py` — the vocabulary

**Problem:** every downstream number depends on counting four cells correctly.

```python
from tmtuning import classification_metrics, confusion_frame
classification_metrics(y_true, y_pred)   # → precision, recall, fpr, f1, alert_rate, ...
```

**The decision it encodes:** when there are no alerts, precision is `NaN`, not
`0.0`. Those are different claims — "undefined" versus "this threshold has zero
precision" — and conflating them corrupts every sweep that runs off the top of a
distribution. `safe_divide` exists solely to enforce that.

It also computes `accuracy`, with a docstring telling you never to quote it.

---

### `data.py` — the populations

**Problem:** you need something to tune against, and the properties of that
something determine what the exercise can teach.

```python
from tmtuning import generate_population, spec_population
```

Two generators, doing two different jobs:

| Generator | Label | Used by |
| --- | --- | --- |
| `spec_population()` | Drawn **independently** of amount — your spec's Week 2 code, verbatim | Week 2 |
| `generate_population()` | **Caused** by behaviour — risk rises with value, velocity, jurisdiction, KYC rating | Weeks 3–10 |

**The decision it encodes:** the risk-linked generator solves for its intercept
by bisection so the case rate hits the prevalence you asked for. Ask for 3% and
you get 3%, whatever else you change — which means a result that moves is a
result about your change, not about the base rate drifting underneath you.

It carries eight fields your Week 10 capstone needs, plus `drift_strength` to
make populations move over time, and `split_by_period()` for out-of-time
validation — splitting on *time*, never at random.

---

### `thresholds.py` — sweeps and selection

**Problem:** finding the right number, under a constraint.

```python
from tmtuning import threshold_sweep, marginal_yield, optimise_threshold
sweep = threshold_sweep(pop, "monthly_wire_value", "case", n_thresholds=40)
best  = optimise_threshold(sweep, objective="recall", max_alerts=1_500)
```

**Two decisions it encodes.**

*Quantile grids, not linear ones.* Candidates come from the data's own
distribution, spanning the 5th to 99.9th percentile with half concentrated above
the 95th. A grid starting at the median silently caps any rule at alerting on
half its population.

*Constrained, not composite.* `optimise_threshold` maximises an objective
*subject to* a capacity limit, rather than maximising F1. F1 prices one missed
case exactly equal to one wasted investigation — which no financial crime
function actually believes, and none would defend in writing.

`marginal_yield()` is the one to reach for when someone asks why not go lower:
it prices only the alerts you are about to add, not the ones you already have.

---

### `segments.py` — per-segment calibration

**Problem:** allocating a fixed alert budget across segments that behave
differently.

```python
from tmtuning import segment_summary, compare_uniform_vs_segmented
compare_uniform_vs_segmented(pop, capacity=2_500)
```

**The decision it encodes:** allocation is done by **Lagrangian sweep over
concave envelopes**, not greedy best-first. That sounds academic; it is not.

A first version ranked raw grid steps by marginal yield and produced allocations
*worse* than a single global threshold — a step of 5 alerts that happened to
contain one case scored 0.20 and outbid a corporate run of 100 alerts containing
8 (0.08). Greedy chased sampling noise.

Taking each segment's upper concave envelope first collapses runs of steps into
their aggregate slope, which removes the noise and makes the allocation provably
optimal over the candidates. There is a regression test asserting segmented
never loses to uniform, across seeds and budgets.

---

### `challenger.py` — comparison and missed-risk

**Problem:** deciding whether a proposed rule is actually better, and bounding
what a change gives up.

```python
from tmtuning import compare_rules, rule_overlap, btl_test, wilson_interval
```

**The decision it encodes:** `rule_overlap()` exists because netted metrics lie
by omission. A challenger that nets +3 cases by losing 40 and gaining 45 is not
a tweak — it is a different rule detecting different behaviour. The overlap
table breaks the netting into four cells so you can see *who* stops being
caught.

`wilson_interval()` rather than the textbook interval, for one specific reason:
with 0 cases in 100 sampled records the Wald interval returns `[0, 0]` — zero
width — implying certainty that no risk sits below the line. Wilson returns
`[0%, 3.7%]`, which across 30,000 records is up to ~1,100 missed cases. Same
data, same review effort, opposite conclusions.

---

### `stability.py` — drift detection

**Problem:** knowing when a threshold has gone stale.

```python
from tmtuning import psi, stability_report
report = stability_report(pop, threshold=50_000, baseline_periods=3)
```

**The decision it encodes:** control limits are anchored to a **baseline
window** — the periods the threshold was tuned on — not to the whole series.
Derive them from everything and a steadily drifting metric inflates its own mean
and standard deviation until the limits chase the drift and nothing ever
breaches. A control chart that cannot fire is assurance theatre.

Volume uses Poisson limits (variance tied to level); precision, a proportion,
uses normal limits.

---

### `scoring.py` — weighted risk scores

**Problem:** moving past a single threshold without losing explainability.

```python
from tmtuning import percentile_score, band_score, weighted_score, decile_table
```

**Two decisions it encodes.**

*Components are ranked before weighting.* Raw pounds beside a raw transaction
count makes the weights decorative — amount decides every alert whatever number
sits next to it. Percentile ranks make them commensurable, and ties take their
average rank so a discrete count does not carry ordering the data cannot support.

*Weights must sum to 1.* `weighted_score` refuses otherwise, because weights
that do not silently rescale the score — and a fixed cut-off then means
something different from one run to the next, with nobody noticing.

---

### `reporting.py` and `decision_log.py` — the deliverables

```python
from tmtuning import mini_tuning_paper, tuning_paper, TuningDecisionLog
```

Two paper generators — the five-section mini paper your spec asks for, and a
fuller one with every piece of supporting evidence. Both generated from the
analysis objects, so they cannot disagree with each other.

**The decision they encode:** omitted evidence becomes a stated limitation
automatically. Pass no below-the-line test and the paper says *"No below-the-line
test was performed, so missed risk below the threshold is unquantified."*
Skipping work costs you a sentence in the document rather than going unnoticed.

`TuningDecisionLog` implements your spec's Recommended Habit — every option
tested, its metrics, operational implications, assumptions and rationale,
**including the ones rejected**. A paper reports what won; the log reports what
lost and why, which is what somebody needs in three years when you are no longer
in the role.

---

## 4. The ten weeks, and what each one proves

Real output from a fixed seed. If your run differs, something has changed.

### Week 1 — the accuracy trap

```
               rule  alerts  tp  fn  accuracy  precision  recall
   Alert on nothing       0   0 305    0.9695        NaN     0.0
Alert on everything   10000 305   0    0.0305     0.0305     1.0
```

A rule that detects nothing scores 97%. Accuracy is dominated by the
true-negative cell — the one cell nobody pays for.

### Week 2 — a flat precision curve is a finding

```
 threshold  alert_count  true_positives  precision  recall
     25000         3268             138     0.0422  0.6540
     50000         1447              64     0.0442  0.3033
     75000          563              25     0.0444  0.1185
    100000          197               9     0.0457  0.0427
```

Precision: 4.22%, 4.42%, 4.44%, 4.57% — flat, at the base rate, while recall
collapses. Your spec's generator draws the label independently of the amount, so
no threshold can separate them. **The flat line is the answer**, and it kills the
rule rather than inviting you to keep sliding the number.

### Week 3 — a recommendation that states its cost

```
  CURRENT:   £50,000
  PROPOSED:  £141,687

  Alert volume      3,159 -> 781
  Precision         11.62% -> 15.11%
  Recall            40.78% -> 13.11%
  Effort per case   8.6 -> 6.6 alerts
```

Note where the recall cost sits: in the recommendation, not an appendix.

### Week 4 — an over-capacity queue never settles

```
  Incumbent £50k    26,762 alerts/month -> backlog after 3 months  61,386, wait 204.6 days
  At capacity        6,300 alerts/month -> backlog after 3 months       0, wait   0.0 days
```

It does not stabilise at a larger backlog — it grows without limit. A 205-day
wait means the funds have moved before anyone opens the alert, so the
incumbent's 40% recall counts alerts nobody reached.

### Week 5 — segmentation must be tested at equal volume

```
                                 approach  alerts  tp  precision  recall
Single threshold £29,841 (volume-matched)    6819 704     0.1032  0.5931
      Retail £20,000 / Corporate £100,000    6819 679     0.0996  0.5720
```

Your spec's segments are assigned by `np.random.rand() > 0.7` — at random. At
equal alert volume the **single threshold wins, 704 cases to 679**. Two
thresholds, twice the governance, negative benefit.

### Week 6 — the operator does the work

```
                                       rule  alerts  tp   fn  precision  recall
                    incumbent: amount > 50k    4705 533  807     0.1133  0.3978
  challenger: amount > 30k AND velocity > 5    1259 247 1093     0.1962  0.1843
   variant: amount > 30k OR velocity > 5      8336 832  508     0.0998  0.6209
variant: 30k AND (velocity > 5 OR risk HIGH)   1828 393  947     0.2150  0.2933
```

AND nearly doubles precision and halves recall. OR is the mirror image. Same two
indicators, opposite operating points.

### Week 7 — read the KS statistic, not the p-value

```
                     comparison  ks_statistic  p_value    psi
               Year 1 vs Year 2        0.0888   0.0000 0.0382
               Year 1 vs Year 3        0.1632   0.0000 0.1348
Year 1 vs itself (random split)        0.0099   0.4448 0.0025
```

Both real comparisons read p = 0.0000 — and so would almost anything at 30,000
records, because KS power grows with *n*. The bottom row is the same
distribution split at random: the **noise floor, 0.0099**. Year 3 clears it by a
factor of 16. That is the evidence.

### Week 8 — a working score has a steep gradient

```
 decile  records  cases  case_rate   lift  cumulative_recall
      1     4497    668     0.1485 4.9929             0.4993
      2     4497    300     0.0667 2.2423             0.7235
      3     4497    145     0.0322 1.0838             0.8318
```

The top decile carries 5× the base-rate risk and half of all cases in 10% of the
population. At matched volume the score finds **698 cases against a single
threshold's 538** — +29.7% for the same investigator effort.

### Week 9 — the paper writes its own limitations

```
Sections: Background, Method, Results, Recommendation, Limitations
```

Generated from the analysis. Give it no below-the-line test and the limitation
appears by itself.

### Week 10 — a constraint is a budget, not a target to undershoot

```
            Metric Before  After   Change
Alerts (per month)  2,241  1,064     -53%
         Precision  7.71% 10.21%  +2.50pp
            Recall 69.93% 43.93% -25.99pp
               FPR 25.58% 11.81% -13.77pp
```

Your spec's *example* calibration cuts volume 94% — and uses only **10% of the
capacity** that was the binding constraint, while recall falls 59 points. The
week records it as rejected with reasons, searches the design space, and
recommends the calibration above instead.

---

## 5. What the build produces

Running all ten weeks writes 13 artefacts to `outputs/`:

| Kind | Count | Files |
| --- | --- | --- |
| Charts | 9 `.png` | Risk yield and precision-recall (wk 3), alert volume against capacity (wk 4, wk 10), segment curves (wk 5), volume and precision control charts (wk 7), yield by decile (wk 8), incumbent stability (wk 10) |
| Papers | 3 `.md` | Full tuning paper and five-section mini paper (wk 9), capstone paper (wk 10) |
| Decision log | 1 `.md` | 16 options tested, including the rejected ones |

Three reference copies are committed in `docs/` so you can read the expected
deliverable without running anything:
`example_tuning_paper.md`, `example_mini_tuning_paper.md`, `example_decision_log.md`.

`outputs/` is gitignored — it regenerates.

---

## 6. Verifying it yourself

```bash
cd projects/tm-rule-tuning
pip install -r requirements.txt

python -m pytest tests/ -q          # 143 tests
python weeks/week01_tuning_framework.py
```

The test suite is worth reading, not just running. It covers three things:

1. **Arithmetic against hand-computed values** — the Week 1 worked example,
   the Wilson interval against published figures (0 successes in 100 trials →
   upper bound 0.0370).
2. **Properties that must hold** — alerts fall monotonically as thresholds rise;
   capacity is never exceeded; decile tables partition the population.
3. **Regressions for bugs found while building** — segment calibration must
   never underperform a single global threshold at the same budget; a weighted
   score must beat a single threshold at matched volume; report tables must
   render integers as integers.

That third category is the useful one to read. Each test names a real defect
that existed and was fixed.

---

## Where to go next

Open `TRAINING_PLAN.md` and start Week 1, Monday. About four hours a week.

Everything in this walkthrough is machinery — the course is where you learn to
*use* it, and more importantly to decide what the numbers do and do not let you
claim.
