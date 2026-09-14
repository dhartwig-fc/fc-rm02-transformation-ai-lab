"""Threshold sweeps, marginal yield and constrained selection."""

import numpy as np
import pandas as pd
import pytest

from tmtuning.data import generate_population
from tmtuning.thresholds import (apply_rule, capacity_frontier, default_thresholds,
                                 marginal_yield, optimise_threshold, threshold_sweep)


@pytest.fixture(scope="module")
def population():
    return generate_population(n=8_000, seed=1, n_periods=6)


@pytest.fixture(scope="module")
def sweep(population):
    return threshold_sweep(population, "monthly_wire_value", "case", n_thresholds=30)


def test_apply_rule_gt_excludes_the_boundary():
    """A rule written '> £10,000' must not alert on exactly £10,000 -- which is
    precisely the amount a customer structuring to the threshold would deposit."""
    values = np.array([9_999.0, 10_000.0, 10_001.0])
    assert apply_rule(values, 10_000, "gt").tolist() == [False, False, True]
    assert apply_rule(values, 10_000, "ge").tolist() == [False, True, True]


def test_apply_rule_rejects_unknown_comparison():
    with pytest.raises(ValueError, match="gt.*ge"):
        apply_rule(np.array([1.0]), 0, "lt")


def test_default_thresholds_reach_below_the_median(population):
    """A grid starting at the median silently caps any rule at a 50% alert rate,
    which breaks segment calibration for high-value segments."""
    values = population["monthly_wire_value"]
    grid = default_thresholds(values, 40)
    assert grid.min() < values.median()
    assert grid.max() <= values.max()
    assert len(grid) > 1


def test_default_thresholds_validates_quantile_range():
    with pytest.raises(ValueError, match="q_low < q_high"):
        default_thresholds(np.arange(100.0), 10, q_low=0.9, q_high=0.1)


def test_alerts_fall_monotonically_as_threshold_rises(sweep):
    assert sweep["alerts"].is_monotonic_decreasing
    assert sweep["tp"].is_monotonic_decreasing
    assert sweep["recall"].is_monotonic_decreasing


def test_sweep_respects_explicit_thresholds(population):
    explicit = [25_000, 50_000, 75_000]
    result = threshold_sweep(population, "monthly_wire_value", thresholds=explicit)
    assert result["threshold"].tolist() == explicit


def test_sweep_rejects_missing_columns(population):
    with pytest.raises(KeyError, match="not found"):
        threshold_sweep(population, "no_such_column")


def test_marginal_precision_is_below_cumulative_precision(sweep):
    """Cumulative precision averages in the productive top of the distribution;
    the marginal figure prices only the alerts about to be added."""
    marginal = marginal_yield(sweep).dropna(subset=["marginal_precision"])
    assert (marginal["marginal_precision"] <= marginal["precision"] + 1e-9).mean() > 0.7


def test_optimise_threshold_respects_capacity(sweep):
    best = optimise_threshold(sweep, objective="recall", max_alerts=500)
    assert best["alerts"] <= 500


def test_optimise_threshold_respects_precision_floor(sweep):
    best = optimise_threshold(sweep, objective="recall", min_precision=0.05)
    assert best["precision"] >= 0.05


def test_optimise_threshold_raises_when_infeasible(sweep):
    with pytest.raises(ValueError, match="No threshold satisfies"):
        optimise_threshold(sweep, min_precision=0.99, max_alerts=1)


def test_optimise_threshold_returns_none_when_not_strict(sweep):
    assert optimise_threshold(sweep, min_precision=0.99, max_alerts=1, strict=False) is None


def test_optimise_threshold_rejects_unknown_objective(sweep):
    with pytest.raises(KeyError, match="not a column"):
        optimise_threshold(sweep, objective="profit")


def test_capacity_frontier_never_exceeds_its_budget(sweep):
    frontier = capacity_frontier(sweep, [200, 400, 800]).dropna(subset=["alerts"])
    assert (frontier["alerts"] <= frontier["capacity"]).all()
    # More budget must never buy less detection.
    assert frontier["tp"].is_monotonic_increasing
