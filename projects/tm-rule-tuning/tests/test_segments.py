"""Segment calibration -- including the regression that motivated the rewrite."""

import numpy as np
import pytest

from tmtuning.data import generate_population
from tmtuning.segments import (calibrate_segments, compare_uniform_vs_segmented, concave_envelope,
                               segment_summary, segment_sweeps, uniform_threshold_at_capacity)


@pytest.fixture(scope="module")
def population():
    return generate_population(n=20_000, seed=505, n_periods=12)


def test_concave_envelope_has_strictly_decreasing_slopes():
    """The envelope is what makes greedy allocation optimal; if its slopes are
    not decreasing it is not an envelope."""
    noisy = [(0.0, 0.0, np.inf), (10.0, 1.0, 90.0), (20.0, 1.0, 80.0),
             (30.0, 5.0, 70.0), (40.0, 5.5, 60.0), (50.0, 9.0, 50.0)]
    hull = concave_envelope(noisy)
    slopes = [(hull[i + 1][1] - hull[i][1]) / (hull[i + 1][0] - hull[i][0])
              for i in range(len(hull) - 1)]
    assert slopes == sorted(slopes, reverse=True)
    assert all(a > b for a, b in zip(slopes, slopes[1:]))


def test_concave_envelope_keeps_endpoints():
    points = [(0.0, 0.0, np.inf), (10.0, 2.0, 50.0), (20.0, 3.0, 40.0)]
    hull = concave_envelope(points)
    assert hull[0][0] == 0.0
    assert hull[-1][0] == 20.0


def test_concave_envelope_drops_dominated_points():
    """A point below the chord between its neighbours carries no useful step."""
    points = [(0.0, 0.0, np.inf), (10.0, 0.0, 50.0), (20.0, 10.0, 40.0)]
    hull = concave_envelope(points)
    assert [p[0] for p in hull] == [0.0, 20.0]


def test_segment_summary_shares_sum_to_one(population):
    summary = segment_summary(population)
    assert summary["share_of_rows"].sum() == pytest.approx(1.0)
    assert summary["share_of_cases"].sum() == pytest.approx(1.0)


def test_segment_sweeps_use_their_own_grids(population):
    """A pooled grid would be dominated by the largest segment."""
    sweeps = segment_sweeps(population, n_thresholds=20)
    assert len(sweeps) > 1
    maxima = {name: sweep["threshold"].max() for name, sweep in sweeps.items()}
    assert max(maxima.values()) > 3 * min(maxima.values())


def test_calibration_never_exceeds_the_budget(population):
    for capacity in [200, 800, 2_000]:
        result = calibrate_segments(population, total_capacity=capacity)
        total = result[result["segment"] == "TOTAL"].iloc[0]
        assert total["alerts"] <= capacity


def test_calibration_rejects_negative_capacity(population):
    with pytest.raises(ValueError, match="non-negative"):
        calibrate_segments(population, total_capacity=-1)


def test_zero_capacity_produces_no_alerts(population):
    total = calibrate_segments(population, total_capacity=0)
    total = total[total["segment"] == "TOTAL"].iloc[0]
    assert total["alerts"] == 0


def test_uniform_baseline_respects_capacity(population):
    result = uniform_threshold_at_capacity(population, capacity=1_000)
    assert result["alerts"].sum() <= 1_000
    assert result["threshold"].nunique() == 1  # one global threshold, by definition


@pytest.mark.parametrize("seed", [505, 42, 7])
@pytest.mark.parametrize("capacity", [500, 1_500, 3_000])
def test_segmented_allocation_beats_uniform(seed, capacity):
    """Regression test for the bug that motivated the concave envelope.

    A greedy allocator ranking raw grid steps by marginal yield chased sampling
    noise and could land *below* a single global threshold at the same budget.
    Segment calibration that loses to the baseline it is meant to improve on is
    worse than not shipping it, so this asserts the direction across seeds and
    budgets rather than trusting one happy example.
    """
    population = generate_population(n=20_000, seed=seed, n_periods=12)
    comparison = compare_uniform_vs_segmented(population, capacity=capacity)
    total = comparison[comparison["segment"] == "TOTAL"].iloc[0]
    assert total["alerts_segmented"] <= capacity
    assert total["tp_segmented"] >= total["tp_uniform"], (
        f"segmented allocation found {int(total['tp_segmented'])} cases vs "
        f"{int(total['tp_uniform'])} for a single global threshold"
    )
