# Transaction Monitoring Rule Tuning

A 10-week practical curriculum in advanced transaction-monitoring rule tuning,
plus the reusable Python toolkit the exercises are built on.

Part of the [Financial Crime AI Lab](../../README.md).

## What this is

Ten runnable weekly exercises that take you from "what is a confusion matrix"
to a complete, validation-ready tuning paper. Every week is a script you run and
read; there are no gaps to fill in from a slide deck.

The curriculum is in **[LEARNING_PLAN.md](LEARNING_PLAN.md)**. Weeks 1-3 follow
a supplied specification; Weeks 4-10 are derived from its stated learning
outcomes and are marked as such.

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
| 3 | Threshold optimisation | Constrained recommendation, out-of-time check |
| 4 | Alert volume and risk yield curves | Three charts, a costed operating point |
| 5 | Segment-specific calibration | Per-segment thresholds, uplift at equal budget |
| 6 | Above and below-the-line testing | Missed-risk estimate with a confidence interval |
| 7 | Stability, instability and drift | Control charts, PSI, monitoring triggers |
| 8 | Incumbent versus challenger | Overlap decomposition, decision gates |
| 9 | Validation-ready tuning papers | A generated tuning paper |
| 10 | Capstone: full backtest and calibration | The complete exercise, end to end |

## The toolkit

`src/tmtuning/` is written to be lifted into real work, not just to serve the
exercises.

| Module | What it does |
| --- | --- |
| `metrics` | Confusion counts and the AML metric set, with honest NaN handling |
| `data` | Synthetic populations — spec-faithful and risk-linked, with optional drift |
| `thresholds` | Quantile sweeps, marginal yield, capacity-constrained selection |
| `segments` | Per-segment calibration; allocates an alert budget to maximise detection |
| `stability` | PSI, period performance, baseline-anchored control limits |
| `challenger` | Champion/challenger comparison, ATL/BTL sampling, Wilson intervals |
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

106 tests. They cover the metric arithmetic against hand-computed values, the
Wilson interval against published figures, and several regressions for bugs
found while building this — notably that segment calibration must never
underperform a single global threshold at the same alert budget.

## A word on the numbers

Every figure these exercises produce comes from a synthetic generator whose
ground truth you were handed. Real tuning is done against a label set that is
sparse, late and partly wrong.

That is the reason the method insists on stating assumptions, bounding what it
cannot measure, and validating out of time — those habits are what transfer.
The numbers do not.
