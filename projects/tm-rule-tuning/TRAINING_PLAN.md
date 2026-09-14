# Weekly Training Plan

A 10-week schedule you can follow session by session. Every session is built
round a worked example with real numbers — run the code, read what it printed,
then do a variation yourself.

**About four hours a week.** [`LEARNING_PLAN.md`](LEARNING_PLAN.md) is the
specification this follows; this document is how you work through it.

---

## The weekly rhythm

| Session | Time | What you do |
| --- | --- | --- |
| **Monday** | 1 hour | Read the idea and the worked example below |
| **Wednesday** | 1 hour | Run the week's script and read its output |
| **Friday** | 1–2 hours | Do the exercise and write down what you found |
| **Weekend** | 30 min | Write a one-page tuning recommendation |

The Friday and weekend sessions are the ones that build the skill. Running a
sweep takes seconds; deciding what the numbers let you claim, and writing it so
somebody else can attack it, is the job.

## Before Week 1

```bash
cd projects/tm-rule-tuning
pip install -r requirements.txt
python -c "import pandas, numpy, sklearn, matplotlib; print('Environment ready')"
```

Keep a file called `notes.md` beside you. Every Friday, write three things into
it: what you ran, what surprised you, what you would tell a risk owner. By Week
10 that file is more useful than anything in this repo.

**All numbers below are real output from fixed random seeds.** If you run the
script and see something different, something has changed — investigate it
rather than assuming you misread.

---

# Week 1 — Building the Tuning Framework

> **The idea:** decide how you will measure success before you change anything.

## Monday (1 hour) — Read

A rule either alerts or it doesn't. A customer either is a case or isn't. That
gives four outcomes, and every metric in tuning is built from counting them.

Take four customers under the rule *alert if monthly cash deposits > £10,000*:

| Customer | Alert | True case | What it is |
| --- | --- | --- | --- |
| A | Yes | Yes | **True positive** — the rule worked |
| B | Yes | No | **False positive** — wasted investigation |
| C | No | Yes | **False negative** — *missed risk* |
| D | No | No | **True negative** — correctly left alone |

From those four counts:

- **Precision** = TP / (TP + FP) — of the alerts we raise, how many are real?
  This is what investigators experience.
- **Recall** = TP / (TP + FN) — of the risk present, how much do we catch?
  This is what a regulator asks about.

Now the trap. Here are two useless rules on a population with a 3% case rate:

```
               rule  alerts  tp  fn  accuracy  precision  recall
   Alert on nothing       0   0 305    0.9695        NaN     0.0
Alert on everything   10000 305   0    0.0305     0.0305     1.0
```

**"Alert on nothing" scores 97% accuracy and detects zero risk.** Accuracy is
dominated by the true-negative cell, which is the one cell nobody pays for. It
would pass an accuracy test and fail the only test that matters. Never put
accuracy in a tuning paper.

The last piece is capacity. Twelve investigators working 8 alerts a day over 21
days is 2,016 alerts a month. That single number turns "what is the best
threshold?" into a question with an answer: *maximise detection, subject to
alerts ≤ 2,016*.

## Wednesday (1 hour) — Run

```bash
python weeks/week01_tuning_framework.py
```

Read every section, but stop at section 4. It prints:

```
[[1 1]
 [1 1]]
```

All four cells are 1, so you cannot tell from the grid which is which. Now read
the line underneath: `TN=1  FP=1  FN=1  TP=1`. **`sklearn` returns
`[[TN, FP], [FN, TP]]` — the top-left is the true negative, not the true
positive.** Published tuning decks get this backwards regularly. Always unpack
with `.ravel()`.

## Friday (1–2 hours) — Do

Build the confusion matrix by hand first, on paper, for this sample:

| Customer | Alert | True case |
| --- | --- | --- |
| A | Yes | Yes |
| B | Yes | No |
| C | Yes | No |
| D | No | Yes |
| E | No | No |

Then check yourself:

```python
from tmtuning import classification_metrics, confusion_frame

y_true = [1, 0, 0, 1, 0]
y_pred = [1, 1, 1, 0, 0]

print(confusion_frame(y_true, y_pred))
print(classification_metrics(y_true, y_pred))
```

You should get TP=1, FP=2, FN=1, TN=1 — precision 33.3%, recall 50%.

Then answer in `notes.md`, in your own words:

1. What does a false negative cost the bank that a false positive does not?
2. Why is a false positive also a *detection* problem, not just a cost one?
3. Your operation has 8 investigators at 20 alerts/day over 21 working days.
   What is the monthly capacity, and what does it let you say to someone
   proposing a threshold that generates 5,000 alerts a month?

## Weekend (30 min) — Write

One page: **"How we will measure TM-014."** Name your label definition
(what counts as a true case, and over what window), the four metrics you will
report, the one you will not report and why, and your monthly capacity figure.

This page is the control against which every later week is judged.

## Check yourself

- Why is precision the metric investigators feel, and recall the metric
  regulators ask about?
- A colleague reports 96% accuracy on a TM rule. What is your first question?
- What is the top-left cell of `sklearn.metrics.confusion_matrix`?

---

# Week 2 — Rule Backtesting Fundamentals

> **The idea:** evaluate the rule you already have, using history.

## Monday (1 hour) — Read

A backtest re-runs a rule over historical data and scores it against what
actually happened. The hard part is not the code — it is the sample.

Five ways a sample lies to you:

| Bias | What it looks like | What it does |
| --- | --- | --- |
| Alert-only sample | Backtest using only records that alerted | Recall becomes undefined, not 100% |
| Investigated-only labels | Treat "no SAR" as "not a case" | Below-the-line risk is invisible |
| Survivorship | Exclude customers exited in the period | Removes your highest-risk customers |
| Single period | One month, extrapolated | Seasonality reads as performance |
| Post-hoc window | Pick the threshold, then the window that supports it | Will not hold out of time |

The second one is the deep one. A SAR can only be filed on a case somebody
investigated, and somebody only investigates what alerted. So SAR labels are
systematically missing below the line, and **measured recall is optimistic by
construction.**

## Wednesday (1 hour) — Run

```bash
python weeks/week02_backtesting_fundamentals.py
```

The week's exercise sweeps four thresholds. Here is the output:

```
 threshold  alert_count  true_positives  precision  recall
     25000         3268             138     0.0422  0.6540
     50000         1447              64     0.0442  0.3033
     75000          563              25     0.0444  0.1185
    100000          197               9     0.0457  0.0427
```

Look at the precision column: **4.22%, 4.42%, 4.44%, 4.57%.** Essentially flat —
and flat at the population base rate. Recall meanwhile collapses from 65% to 4%.

That is not a tuning failure. In this generator the case label is drawn
independently of the amount:

```python
"case": np.random.binomial(1, 0.04, n)
```

Risk and transaction value are statistically independent, so no threshold on
amount can separate them. **The flat line is the finding.** Recognising it is
the skill: a sweep with no precision lift means the variable carries no risk
signal, which kills the rule — it does not invite you to keep sliding the number
until one looks acceptable.

## Friday (1–2 hours) — Do

Run the same sweep on a population where risk and value *are* related, and see
the difference:

```python
from tmtuning import generate_population, threshold_sweep, show

realistic = generate_population(n=20_000, seed=42)
sweep = threshold_sweep(realistic, "monthly_wire_value", "case",
                        thresholds=[25_000, 50_000, 75_000, 100_000])
show(sweep[["threshold", "alerts", "tp", "precision", "recall"]])
```

Precision should now **rise** as the threshold tightens. Write in `notes.md`:

1. What is the precision at each threshold, and by how much does it rise?
2. At a capacity of 2,000 alerts, which threshold would you choose, and why?
3. Someone shows you a flat precision curve on real data. Give three possible
   causes and the different action each one demands.

## Weekend (30 min) — Write

One page: **"Sample design for the TM-014 backtest."** State your period span,
what you include and exclude, how you treat exited customers, and which of the
five biases above you cannot fully rule out — with the direction each one pushes
your numbers.

## Check yourself

- Why does an alert-only sample make recall undefined rather than 100%?
- Precision is flat across every threshold you test. Is that a bad backtest or a
  result?
- In which direction does SAR-label sparsity bias measured recall?

---

# Week 3 — Threshold Optimisation

> **The idea:** pick the number with evidence, not judgement alone.

## Monday (1 hour) — Read

Judgement is good at things evidence is bad at: knowing a typology exists,
knowing a threshold sits on a round number people structure beneath. It is
unreliable at one specific thing — estimating how a change in a number moves
volume and detection, because both depend on the shape of a distribution nobody
can hold in their head.

So: judgement sets the candidate range and vetoes nonsense. Evidence picks the
point inside it.

Two concepts do most of the work this week.

**Marginal precision.** Headline precision is *cumulative* — it averages in the
productive top of the distribution. Marginal precision prices only the alerts
you are about to add. It is always worse, and it is the honest number. When it
falls to the population base rate, your next tranche of alerts is no better than
picking customers at random.

**Sensitivity.** Move a threshold ±20% and see how far volume moves. In the
week's run, a ±20% move around £50,000 swings alert volume across a **44%**
range while precision barely moves. Volume follows the density of the
distribution; precision follows a much flatter risk gradient. That is why a
threshold rounded to a comfortable number can still be wrong by a third of the
operation's workload.

## Wednesday (1 hour) — Run

```bash
python weeks/week03_threshold_optimisation.py
```

Section 2 runs the spec's fixed sweep, `range(10000, 120000, 5000)`. Section 4
rebuilds the grid from the data's own quantiles and explains why: a £5k step
spends 22 candidates evenly across a range the data does not occupy evenly.
Roughly half the customers sit below £10k and are never examined.

Section 9 is the deliverable — the shape every recommendation you write should
take:

```
  CURRENT:   £50,000
  PROPOSED:  £141,687

  Alert volume      3,159 -> 781
```

Read the four numbered rationale points under it. Notice that the recommendation
**states what it accepts**: recall falls from 38.4% to 11.7%, and that appears
in the recommendation rather than an appendix.

## Friday (1–2 hours) — Do

Run your own sensitivity analysis and find where marginal yield dies:

```python
from tmtuning import (generate_population, threshold_sweep, marginal_yield,
                      optimise_threshold, show)

pop = generate_population(n=40_000, seed=2024, n_periods=12)
sweep = threshold_sweep(pop, "monthly_wire_value", "case", n_thresholds=40)

show(marginal_yield(sweep)[["threshold", "precision", "marginal_precision"]])
print("base rate:", round(pop["case"].mean(), 4))

best = optimise_threshold(sweep, objective="recall", max_alerts=1_500)
print(best[["threshold", "alerts", "precision", "recall"]])
```

In `notes.md`:

1. At roughly which threshold does marginal precision fall to the base rate?
2. Compare `precision` and `marginal_precision` in the same row. Which is
   higher, always, and why?
3. Re-run `optimise_threshold` with `objective="precision"` and with
   `objective="f1"`. The three answers disagree. Which would you defend to a
   risk owner, and what is wrong with the other two?

## Weekend (30 min) — Write

One page: **"TM-014 threshold recommendation."** Use the Week 3 deliverable
structure — current, proposed, evidence in four points (constraint, marginal
yield, sensitivity, out-of-time), and an explicit statement of what the change
accepts.

## Check yourself

- Why is cumulative precision always at least as good as marginal precision?
- What is wrong with maximising F1 as a tuning objective?
- Why quantile-spaced candidates rather than evenly spaced ones?

---

# Week 4 — Alert Volume and Capacity Modelling

> **The idea:** the statistically best threshold may be operationally impossible.

## Monday (1 hour) — Read

The arithmetic is simple:

```
capacity = 12 investigators × 25 alerts/day = 300 per day
         = 300 × 21 working days           = 6,300 per month
```

Every assumption inside that is optimistic — no leave, no sickness, no training,
no QA sampling, every alert costing the same to review, investigators
interchangeable. Use the headline for the arithmetic, then state the haircut you
applied and why. A capacity number quoted without its assumptions is the most
common way a tuning paper is made to balance.

Now the part people get wrong. When a rule runs over capacity, the queue does
**not** settle at a larger backlog. It grows without limit, every day, for as
long as the breach continues.

## Wednesday (1 hour) — Run

```bash
python weeks/week04_capacity_modelling.py
```

Section 2 shows the incumbent at **4.2× capacity**. Section 6 simulates what
that does:

```
  Incumbent £50k    26,762 alerts/month -> backlog after 3 months  61,386, wait 204.6 days
  At capacity        6,300 alerts/month -> backlog after 3 months       0, wait   0.0 days
```

A 205-day wait means that by the time an investigator opens the alert, the funds
have moved and the SAR is late. **Detection that arrives too late to act on is
not detection** — so the incumbent's 40% recall is a paper figure counting
alerts nobody reached.

Section 4 has a second lesson. Grid search found £178,881 giving 4,341 alerts —
only **69% of capacity**, leaving nearly a third of the team's month idle. The
threshold producing *exactly* capacity is a quantile lookup, not a search:

```
  Exact capacity threshold: £148,282
    alerts     6,300 (100.0% of capacity)
```

Use a grid to understand the curve's shape; use the quantile to set the number.

## Friday (1–2 hours) — Do

Model your own operation:

```python
import numpy as np
from tmtuning import generate_population, classification_metrics, apply_rule

pop = generate_population(n=250_000, seed=404, n_periods=1)

investigators, per_day, working_days = 8, 20, 21
capacity = investigators * per_day * working_days
print("monthly capacity:", capacity)

# The threshold that produces exactly capacity, by quantile lookup
cut = np.quantile(pop["monthly_wire_value"], 1 - capacity / len(pop))
print("threshold:", round(cut))
print(classification_metrics(pop["case"], apply_rule(pop["monthly_wire_value"], cut)))
```

In `notes.md`:

1. What threshold fits your capacity, and what recall does it buy?
2. Apply a 25% haircut to capacity. How far does the threshold move?
3. Your rule currently runs at 3× capacity. Write the two sentences you would
   say to an operations lead — one about the queue, one about what their
   reported recall actually means.

## Weekend (30 min) — Write

One page: **"TM-014 capacity assessment."** State the capacity and its
assumptions, the current position against it, the queue consequence if nothing
changes, and — this is the part people omit — what additional capacity would
buy, so the resourcing option is visible rather than silently assumed away.

## Check yourself

- Why does an over-capacity queue grow without limit rather than stabilising?
- Why quote alerts per investigator per day rather than alerts per month?
- Your recommendation uses 65% of capacity. Why is that a problem?

---

# Week 5 — Segment-Based Calibration

> **The idea:** one threshold rarely fits all customer types — but prove it.

## Monday (1 hour) — Read

£50,000 a month is extraordinary for a retail customer and unremarkable for a
corporate. One number cannot be right for both, and the cost is paid twice:
retail risk goes undetected while corporate alerts flood the queue with normal
business.

That is the theory. This week is mostly about how to *test* it honestly, because
segmentation is very easy to make look good by accident.

The rule: **compare at equal alert volume.** A segmented rule scored against an
untuned single threshold firing fewer alerts is not a comparison, it is a rigged
one.

## Wednesday (1 hour) — Run

```bash
python weeks/week05_segment_calibration.py
```

Section 1 runs the spec's exercise. The segments come from:

```python
df["segment"] = np.where(np.random.rand(len(df)) > 0.7, "Corporate", "Retail")
```

That assigns the label **at random**, with no reference to any customer
attribute — so the two segments have near-identical base rates and median
amounts. At first the segmented thresholds look better: more alerts, more cases.
Then the fair test:

```
Both spending the same 6,819 alerts
                                 approach  alerts  tp  precision  recall
Single threshold £29,841 (volume-matched)    6819 704     0.1032  0.5931
      Retail £20,000 / Corporate £100,000    6819 679     0.0996  0.5720
```

**At equal volume the single threshold wins — 704 cases against 679.** Two
thresholds, twice the governance, negative benefit.

A segmentation uncorrelated with behaviour cannot improve detection however its
thresholds are set. Sections 2 onward repeat the exercise on segments that carry
real signal, where the uplift is genuine.

## Friday (1–2 hours) — Do

Profile before you segment, then test at equal budget:

```python
from tmtuning import (generate_population, segment_summary,
                      compare_uniform_vs_segmented, show)

pop = generate_population(n=40_000, seed=505, n_periods=12)
show(segment_summary(pop).reset_index())

cmp = compare_uniform_vs_segmented(pop, capacity=2_500)
show(cmp[["segment", "threshold_uniform", "tp_uniform",
          "threshold_segmented", "tp_segmented", "uplift_tp"]])
```

In `notes.md`:

1. Which segment has the highest `case_concentration`, and what does a value
   above 1 mean?
2. What is the total detection uplift at equal alert volume?
3. Segmentation makes segment assignment an AML control. Name two things that
   must now be true about your segment data, and who owns each.

## Weekend (30 min) — Write

One page: **"Should TM-014 be segmented?"** Include the profile, the uplift at
equal budget, the governance cost, and a recommendation. *"No uplift, do not
segment"* is a legitimate and valuable answer — write it that way if that is
what the data says.

## Check yourself

- Why must segmentation be compared at equal alert volume?
- What does it mean for a segmentation to be "uncorrelated with behaviour"?
- What new control does a segmented rule create?

---

# Week 6 — Challenger Rule Development

> **The idea:** propose alternative logic, and find out what it stops catching.

## Monday (1 hour) — Read

A challenger changes the rule's *logic*, not just its number. The spec's
challenger is:

```
Incumbent:   amount > £50k
Challenger:  amount > £30k  AND  velocity > 5 transactions
```

Before combining indicators, test each on its own. An indicator that does not
separate risk will not start doing so because it has been ANDed to one that
does. Look for **lift** (how much more likely a flagged record is to be a case)
*and* **coverage** (how much of the population it flags). An indicator with lift
of 10 that fires on 0.1% of customers cannot move recall however you combine it.

## Wednesday (1 hour) — Run

```bash
python weeks/week06_challenger_rules.py
```

Section 3 scores four designs on the same sample:

```
                                                 rule  alerts  tp   fn  precision  recall
                              incumbent: amount > 50k    4705 533  807     0.1133  0.3978
            challenger: amount > 30k AND velocity > 5    1259 247 1093     0.1962  0.1843
                variant: amount > 30k OR velocity > 5    8336 832  508     0.0998  0.6209
variant: amount > 30k AND (velocity > 5 OR risk HIGH)    1828 393  947     0.2150  0.2933
```

The challenger nearly **doubles precision** (11.3% → 19.6%) and roughly **halves
recall** (39.8% → 18.4%). Same two indicators with OR instead of AND gives the
mirror image. **The combination operator, not the thresholds, is doing most of
the work.**

Now section 4, which is the one that decides the recommendation. The netted
figures hide *who* the challenger stops catching — and it turns out to be
high-value, low-velocity customers. That is a coherent typology, not a random
sample of the incumbent's catch. A challenger that systematically drops one
typology has narrowed coverage, whatever its precision does.

## Friday (1–2 hours) — Do

Design your own challenger and decompose it:

```python
import numpy as np
from tmtuning import (generate_population, compare_rules, rule_overlap,
                      apply_rule, show)

pop = generate_population(n=40_000, seed=808, n_periods=12)
amount, velocity = pop["monthly_wire_value"], pop["velocity"]

rules = {
    "incumbent": np.asarray(apply_rule(amount, 50_000)),
    "mine": np.asarray((amount > 30_000) & (velocity > 4)),
}
show(compare_rules(pop, rules, champion="incumbent").reset_index())
show(rule_overlap(pop, rules["incumbent"], rules["mine"]))
```

In `notes.md`:

1. How many cases does your challenger newly *miss*, and how many does it newly
   catch? (Read the "Champion only" and "Challenger only" rows.)
2. What do the newly-missed cases have in common?
3. Your challenger depends on `velocity`. If that field is null for 30% of
   customers in production, what happens — and to which customers?

## Weekend (30 min) — Write

One page: **"Does the challenger merit further validation?"** Give the three
metrics the spec asks for (precision, recall, volume), the overlap decomposition,
and a verdict with conditions. *"Merits validation"* is not *"adopt"* — it means
the evidence justifies a parallel run.

## Check yourself

- Why does AND versus OR matter more than the thresholds?
- Why is a net gain of +3 cases not enough to recommend a challenger?
- What does "fails closed" mean, and why is it worse than it sounds?

---

# Week 7 — Stability and Drift Testing

> **The idea:** a rule that works in one period may fail later.

## Monday (1 hour) — Read

A threshold is tuned against one snapshot. Behaviour then moves — inflation, a
new product, an acquisition, a typology shift — and a threshold defensible in
January quietly stops being defensible by September.

Three things to tell apart, because they look identical in a monitoring pack and
need opposite responses:

| Pattern | PSI / KS | Volume | Precision | What to do |
| --- | --- | --- | --- | --- |
| **Population drift** | Rises | Moves | Roughly held | Re-tune the number |
| **Seasonality** | Rises and returns | Cyclical | Roughly held | Do *not* re-tune |
| **Rule decay** | Flat | Flat | Falls | Redesign the rule |

Re-tuning in response to a seasonal peak bakes the peak into the baseline and
under-alerts all year. Dismissing real drift as "just seasonal" leaves the rule
stale. The test is simple: **compare like calendar months, never consecutive
ones.**

## Wednesday (1 hour) — Run

```bash
python weeks/week07_stability_and_drift.py
```

Section 3 runs the KS test:

```
                     comparison  ks_statistic  p_value    psi
               Year 1 vs Year 2        0.0888   0.0000 0.0382
               Year 1 vs Year 3        0.1632   0.0000 0.1348
Year 1 vs itself (random split)        0.0099   0.4448 0.0025
```

Look at the p-value column. Years 1 vs 2 and 1 vs 3 both read **0.0000** — but
so would almost any comparison at this sample size, because KS power grows with
*n*. On 30,000 records a significant p-value tells you the samples are not
literally identical, which was never in doubt.

**Read the statistic.** The bottom row is the control: the same distribution
split at random, giving 0.0099. That is the noise floor. Year 3 at 0.1632 clears
it by a factor of 16 — *that* is the evidence, not the p-value.

Section 4 separates seasonality from drift by reading a month × year table down
the columns and across the rows.

## Friday (1–2 hours) — Do

Detect drift in a population you have not been told about:

```python
from tmtuning import generate_population, stability_report, show

# One of these drifts. Run both before reading drift_strength.
a = generate_population(n=30_000, seed=11, n_periods=12, drift_strength=0.0)
b = generate_population(n=30_000, seed=11, n_periods=12, drift_strength=1.6)

for name, pop in [("A", a), ("B", b)]:
    rep = stability_report(pop, 50_000, baseline_periods=3)
    print(f"--- {name}: {int(rep['any_breach'].sum())} of {len(rep)} periods breach")
    show(rep[["period", "alerts", "precision", "psi", "band", "any_breach"]])
```

In `notes.md`:

1. Which population is drifting, and which three signals told you?
2. Is it population drift or rule decay? How do you know?
3. Pick two mitigants from the week's section 8 and say what each would cost you.

## Weekend (30 min) — Write

One page: **"TM-014 monitoring specification."** For each metric: the trigger
level, the frequency, and — the part that is always missing — the **named
action** and its owner. A threshold with no action attached gets noted and moved
past for four quarters running.

## Check yourself

- Why is the KS p-value near-useless at large sample sizes?
- Why must a same-distribution control use a *random* split, not first-half
  versus second-half?
- Volume is up, PSI is up, precision held. Drift or decay?

---

# Week 8 — Advanced Calibration Using Scoring

> **The idea:** move beyond a single threshold.

## Monday (1 hour) — Read

A binary rule forces every indicator to a hard cut. A score keeps the gradient:
a customer just under the amount cut but well over on velocity, in a high-risk
jurisdiction, can still surface.

The spec's scoring rule:

```python
score = amount_score * 0.5 + velocity_score * 0.3 + geo_score * 0.2
```

**The components must be made commensurable before they are weighted.** Raw
pounds beside a raw transaction count makes the weights decorative — amount
would decide every alert whatever number sat next to it. So amount and velocity
become *percentile ranks* (0–100), and the jurisdiction tier becomes an explicit
lookup, because a tier is an ordered judgement rather than a measurement.

The weights are the model. Each needs a sentence somebody will defend — and
*"0.5 performed best"* is circular when the weights were chosen on the same data
the performance was measured on.

## Wednesday (1 hour) — Run

```bash
python weeks/week08_scoring_calibration.py
```

Section 4 is the most informative table a score produces:

```
 decile  records  cases  case_rate   lift  cumulative_recall
      1     4497    668     0.1485 4.9929             0.4993
      2     4497    300     0.0667 2.2423             0.7235
      3     4497    145     0.0322 1.0838             0.8318
      4     4498     69     0.0153 0.5156             0.8834
```

**The top decile holds 5× the base-rate risk and captures half of all cases in
10% of the population.** A steep, monotone gradient is what a working score
looks like. A flat table means the score is not separating risk, however good
its headline precision looks.

Section 6b is the fair test — score against single threshold at matched volume:

```
                  approach  alerts  tp  precision  recall
Single threshold > £50,000    4788 538     0.1124  0.4021
    Weighted score >= 75.9    4788 698     0.1458  0.5217
```

**+160 cases (+29.7%) for the same investigator effort.**

## Friday (1–2 hours) — Do

Build a score with your own weights and see what changes:

```python
from tmtuning import (generate_population, percentile_score, band_score,
                      weighted_score, decile_table, show)

pop = generate_population(n=40_000, seed=808, n_periods=12)
GEO = {"DOMESTIC": 0, "STANDARD": 40, "ELEVATED": 75, "HIGH": 100}

pop["amount_score"] = percentile_score(pop["monthly_wire_value"])
pop["velocity_score"] = percentile_score(pop["velocity"])
pop["geo_score"] = band_score(pop["jurisdiction_risk"], GEO)

# Try (0.5, 0.3, 0.2), then (0.3, 0.5, 0.2), then (0.6, 0.2, 0.2)
pop["score"] = weighted_score({
    "amount": (pop["amount_score"], 0.5),
    "velocity": (pop["velocity_score"], 0.3),
    "geo": (pop["geo_score"], 0.2),
})
show(decile_table(pop)[["decile", "case_rate", "lift", "cumulative_recall"]])
```

In `notes.md`:

1. How much does the top decile's lift change when you shift weight from amount
   to velocity?
2. Try weights that sum to 1.2. What happens, and why is that protection useful?
3. A score has four sets of choices — components, weights, combination method,
   cut-off. A sweep justifies only the last. Write one defending sentence for
   each of your three weights.

## Weekend (30 min) — Write

One page: **"Proposed score cut-off for TM-014."** Give the cut-off, the decile
it falls in, the matched-volume comparison against the incumbent, the observed
probability with its interval, and what the change accepts.

## Check yourself

- Why percentile ranks rather than raw values?
- Why must weights sum to 1?
- What does a flat decile table tell you?

---

# Week 9 — Validation and Governance

> **The idea:** a tuning exercise only succeeds if it is defensible.

## Monday (1 hour) — Read

The analysis is not the deliverable. A tuning change is approved or rejected on
a document, and analysis that cannot be written up in the expected shape does
not get implemented.

Independent validation checks ten areas, and **only two concern the threshold
itself.** The most common reason a paper comes back is not a wrong number — it
is an unstated label definition or an unquantified below-the-line position.

The counter-intuitive part: **a paper with a substantial limitations section is
treated as more credible, not less.** A validator's job is to find what you
missed. If you have named the weaknesses, their findings become confirmations
and the conversation moves to whether the residual risk is acceptable — which is
a decision the risk owner can actually take.

Compare:

> *Weak:* "Labels may be incomplete."
>
> *Strong:* "SAR labels are absent below the line by construction, which biases
> measured recall upward. A 1,500-record sample bounds the missed-case rate at
> 2.4% with 95% confidence, and the recommendation is made against that upper
> bound rather than the point estimate."

Same weakness. The second one closes it.

## Wednesday (1 hour) — Run

```bash
python weeks/week09_tuning_papers.py
```

Section 4a generates the five-section mini paper the spec asks for — Background,
Method, Results, Recommendation, Limitations. Open it:

```bash
cat outputs/week09_mini_tuning_paper.md
```

Then read section 5. The generator is given *no* below-the-line test, and this
appears automatically in the limitations:

> *No below-the-line test was performed, so missed risk below the threshold is
> unquantified.*

**Skipping work costs you a sentence in the document rather than going
unnoticed.** That is the argument for generating the paper from the analysis
instead of typing it.

## Friday (1–2 hours) — Do

Generate a paper for your own analysis:

```python
from tmtuning import (generate_population, threshold_sweep, optimise_threshold,
                      mini_tuning_paper, save_paper)

pop = generate_population(n=20_000, seed=909, n_periods=12)
sweep = threshold_sweep(pop, "monthly_wire_value", "case", n_thresholds=30)
current = threshold_sweep(pop, "monthly_wire_value", "case",
                          thresholds=[50_000]).iloc[0]
proposed = optimise_threshold(sweep, objective="recall", max_alerts=1_200)

paper = mini_tuning_paper(
    rule_name="TM-014", current_rule="ALERT IF monthly_wire_value > 50000",
    population=pop, sweep=sweep, proposed=proposed, current=current,
    author="your name",
    limitations=["Add the one YOU know is weakest about this analysis."],
)
print(save_paper("outputs/my_paper.md", paper))
```

Then rehearse independent challenge. For each of these, write your answer:

1. Why this threshold and not the one either side of it?
2. What is your label definition, and how sparse is it?
3. What risk does this change accept — as a number?
4. Would it hold on data you did not tune on?
5. How will you know when it stops working?
6. **What would change your recommendation?**

Question 6 is the revealing one. If nothing would, the recommendation was not
derived from the evidence.

## Weekend (30 min) — Write

Take your generated paper and add the governance wrapper by hand: who owns the
rule, who owns the threshold, how the change reaches production, what evidence
is retained and where, and when it is next reviewed.

## Check yourself

- Why does a longer limitations section make a paper *more* credible?
- What must survive so somebody can reproduce your result in three years?
- What does "independent" mean in independent challenge?

---

# Week 10 — Capstone

> **The idea:** run the whole thing, end to end, and decide.

## Monday (1 hour) — Read

The brief:

> Existing rule: **cash deposits > £10k in 30 days**, over 100,000 customers.
>
> Constraints: investigation capacity fixed · senior management expects volume
> reduction · missed-risk tolerance limited.

Six steps: baseline, threshold sweep, segment calibration, challenger design,
stability testing, recommendation.

One warning before you start. `Alert Outcome` in the data is a function of the
*incumbent* rule — every record below £10k reads "NO ALERT" because nobody
looked, not because it was reviewed and cleared. Treating that as evidence of no
risk is the easiest way to make a capstone answer look better than it is.

## Wednesday (1 hour) — Run

```bash
python weeks/week10_capstone.py
```

The baseline:

```
    Alert volume   26,897 over 12 periods (2,241 per month)
    Precision      7.71%
    Recall         69.93%
    FPR            25.58%
```

Against a 1,500/month allocation that is 149% of capacity — so volume must come
down. Now read **section 6a carefully**, because it is the week's real lesson.
The spec offers an example calibration (Retail £15k / Corporate £35k /
velocity > 3). Evaluated honestly it cuts volume by 94% — and uses **only 10% of
the capacity** the bank is already paying for, while recall falls 59 percentage
points.

It satisfies "volume reduction" by massively overshooting it, and fails
"missed-risk tolerance limited". **A constraint is a budget, not a target to
undershoot.**

Section 6b searches the design space properly and lands here:

```
            Metric Before  After   Change
Alerts (per month)  2,241  1,064     -53%
         Precision  7.71% 10.21%  +2.50pp
            Recall 69.93% 43.93% -25.99pp
               FPR 25.58% 11.81% -13.77pp
```

Two details worth pausing on. Designs are sized on the **Q4 run-rate**, not the
annual average — 10 of 85 designs average inside capacity across the year but
breach it in Q4. And the segmented option beat the single threshold by 0.10pp,
which is inside noise, so the recommendation **simplifies to one threshold**.

## Friday (1–2 hours) — Do

Work the six steps yourself, and record every option in a decision log:

```python
from tmtuning import (generate_population, classification_metrics, apply_rule,
                      TuningDecisionLog, show)

pop = generate_population(n=100_000, seed=1010, n_periods=12, drift_strength=1.2)
log = TuningDecisionLog(rule="TM-021 Cash Deposit Structuring", author="your name")

for t in [10_000, 15_000, 20_000, 25_000, 30_000]:
    m = classification_metrics(pop["case"], apply_rule(pop["monthly_cash_deposits"], t))
    log.record("Threshold sweep", f"cash > £{t:,}", m,
               operational_implication=f"{m['alerts'] / 12:,.0f} alerts/month",
               outcome="carried forward" if m["alerts"] / 12 <= 1_500 else "rejected",
               rationale="...")

show(log.to_frame()[["option", "alerts", "precision", "recall", "outcome"]])
print(log.save("outputs/my_decision_log.md"))
```

In `notes.md`:

1. Which thresholds fit the 1,500/month allocation?
2. Does adding a velocity condition help at matched volume, or just cut volume?
3. Your recommendation reduces recall. Write the sentence you would put in the
   *first paragraph* stating what it accepts.

## Weekend (30 min) — Write

The full deliverable: a mini tuning paper plus the decision log. Compare yours
against [`docs/example_tuning_paper.md`](docs/example_tuning_paper.md) and
[`docs/example_decision_log.md`](docs/example_decision_log.md) — not to match
them, but to see what you left out.

## Check yourself

- Why is a calibration using 10% of capacity a *failure*, not a conservative
  success?
- Why size on Q4 rather than the twelve-month average?
- Management asked for volume reduction and you delivered it. Why is the paper
  not finished?

---

## After Week 10

**Keep the decision log.** It is the cheapest habit here and the one still
earning its keep in three years, when somebody asks why the threshold is
£20,000 and you are no longer in the role.

**Do the supplementary week.** `extras/atl_btl_testing.py` covers below-the-line
sampling — normally the first evidence a validator asks for, and what turns a
limitations section from an assertion into a measured bound.

**Then the stretch goals** in `LEARNING_PLAN.md`. Two of the seven (PSI and
champion-challenger) you have already done.

### A closing caution

Every number in this plan came from a synthetic generator whose ground truth you
were handed. Real tuning runs against labels that are sparse, late and partly
wrong.

That is exactly why the method's discipline — stating assumptions, bounding what
you cannot measure, validating out of time, writing down what you rejected — is
the part that transfers. The numbers do not.
