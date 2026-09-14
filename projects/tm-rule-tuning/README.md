# Transaction Monitoring Rule Tuning

A 10-week practical curriculum in advanced transaction-monitoring rule tuning,
plus the reusable Python toolkit the exercises are built on.

Part of the [Financial Crime AI Lab](../../README.md).

## What this is

Ten runnable weekly exercises that take you from "what is a confusion matrix"
to a complete, validation-ready tuning paper. Every week is a script you run and
read; there are no gaps to fill in from a slide deck.

The curriculum is in **[LEARNING_PLAN.md](LEARNING_PLAN.md)**, transcribed from
a supplied specification.

## Quick start

```bash
cd projects/tm-rule-tuning
pip install -r requirements.txt

# Check the environment (Week 1 does this too)
python -c "import pandas, numpy, sklearn, matplotlib; print('Environment ready')"

# Work through the weeks in order
python weeks/week01_tuning_framework.py
```

Each week runs standalone and prints its own working. Charts and generated
papers are written to `outputs/`.

The week files are [Jupyter percent-format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html)
scripts: they run as plain Python *and* open as notebooks. To get `.ipynb`
files:

```bash
pip install jupytext
jupytext --to notebook weeks/*.py
```

## The weeks

| Week | Topic | Produces |
| --- | --- | --- |
| 1 | Building the tuning framework | Metric vocabulary, the accuracy trap |
| 2 | Rule backtesting fundamentals | Coverage analysis, selection-bias checklist |
| 3 | Threshold optimisation | Sensitivity analysis, a justified recommendation |
| 4 | Alert volume & capacity modelling | Queue dynamics, a capacity-feasible threshold |
| 5 | Segment-based calibration | Per-segment thresholds, uplift at equal budget |
| 6 | Challenger rule development | Overlap decomposition, a validation verdict |
| 7 | Stability & drift testing | KS tests, control charts, mitigants |
| 8 | Advanced calibration using scoring | Weighted score, deciles, a justified cut-off |
| 9 | Validation & governance | A mini tuning paper and a full one |
| 10 | Capstone backtest & calibration | The complete engagement, end to end |

### Supplementary

`extras/atl_btl_testing.py` — above and below-the-line testing. Not part of the
specification, but below-the-line sampling is normally the first evidence a
validator asks for, and it turns Week 9's limitations section from an assertion
into a measured bound. Run it after Week 6.

## The toolkit

`src/tmtuning/` is written to be lifted into real work, not just to serve the
exercises.

| Module | What it does |
| --- | --- |
| `metrics` | Confusion counts and the AML metric set, with honest NaN handling |
| `data` | Synthetic populations — spec-faithful and risk-linked, with optional drift |
| `teaching` | Presentation helpers for the weekly scripts (banners, tables) |
| `thresholds` | Quantile sweeps, marginal yield, capacity-constrained selection |
| `segments` | Per-segment calibration; allocates an alert budget to maximise detection |
| `stability` | PSI, period performance, baseline-anchored control limits |
| `challenger` | Incumbent/challenger comparison, ATL/BTL sampling, Wilson intervals |
| `scoring` | Weighted risk scores, deciles, probability calibration, cut-offs |
| `plots` | Volume, yield, trade-off, segment and control charts |
| `reporting` | Generates a tuning paper from analysis objects |

```python
from tmtuning import generate_population, threshold_sweep, optimise_threshold

population = generate_population(n=40_000, seed=42)
sweep = threshold_sweep(population, "monthly_wire_value", "case")

# Maximise detection subject to what operations can actually work
best = optimise_threshold(sweep, objective="recall", max_alerts=2_000)
print(f"£{best['threshold']:,.0f} -> {best['recall']:.1%} recall at {best['precision']:.2%} precision")
```

## Tests

```bash
python -m pytest tests/ -q
```

133 tests. They cover the metric arithmetic against hand-computed values, the
Wilson interval against published figures, and several regressions for bugs found
while building this — notably that segment calibration must never underperform a
single global threshold at the same alert budget, and that a weighted score must
beat a single threshold at matched alert volume.

## A word on the numbers

Every figure these exercises produce comes from a synthetic generator whose
ground truth you were handed. Real tuning is done against a label set that is
sparse, late and partly wrong.

That is the reason the method insists on stating assumptions, bounding what it
cannot measure, and validating out of time — those habits are what transfer.
The numbers do not.
