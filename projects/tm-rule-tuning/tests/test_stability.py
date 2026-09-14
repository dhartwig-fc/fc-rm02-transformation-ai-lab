"""PSI, control limits and drift detection."""

import numpy as np
import pytest

from tmtuning.data import generate_population
from tmtuning.stability import (control_limits, period_performance, psi, psi_band, psi_by_period,
                                stability_report)


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=5_000)
    assert psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    baseline = rng.normal(size=20_000)
    shifts = [psi(baseline, rng.normal(loc=delta, size=20_000)) for delta in (0.1, 0.5, 1.0)]
    assert shifts == sorted(shifts)
    assert shifts[0] < 0.1 < shifts[-1]


def test_psi_rejects_empty_samples():
    with pytest.raises(ValueError, match="non-empty"):
        psi(np.array([]), np.array([1.0]))


def test_psi_handles_a_vanished_region():
    """An empty bin is floored, not dropped -- dropping it would understate the
    shift exactly when part of the distribution has disappeared."""
    baseline = np.linspace(0, 100, 1_000)
    truncated = np.linspace(0, 50, 1_000)
    assert psi(baseline, truncated) > 0.25


@pytest.mark.parametrize("value,expected", [
    (0.05, "stable"), (0.15, "moderate shift"), (0.40, "significant shift"),
])
def test_psi_band_labels(value, expected):
    assert psi_band(value) == expected


def test_poisson_limits_scale_with_the_square_root_of_the_level():
    limits = control_limits([100, 100, 100], k=2.0, method="poisson")
    assert limits["centre"] == pytest.approx(100)
    assert limits["upper"] == pytest.approx(100 + 2 * 10)


def test_normal_limits_use_the_sample_spread():
    limits = control_limits([10, 12, 14], k=1.0, method="normal")
    assert limits["centre"] == pytest.approx(12)
    assert limits["upper"] == pytest.approx(12 + np.std([10, 12, 14], ddof=1))


def test_lower_limit_is_floored_at_zero():
    assert control_limits([1, 2, 3], k=10.0)["lower"] == 0.0


def test_control_limits_rejects_unknown_method():
    with pytest.raises(ValueError, match="normal.*poisson"):
        control_limits([1, 2, 3], method="student")


def test_period_performance_covers_every_period():
    population = generate_population(n=6_000, seed=3, n_periods=6)
    result = period_performance(population, 50_000)
    assert len(result) == population["period"].nunique()


def test_baseline_must_leave_something_to_monitor():
    population = generate_population(n=6_000, seed=3, n_periods=6)
    with pytest.raises(ValueError, match="whole series"):
        stability_report(population, 50_000, baseline_periods=6)


def test_drift_breaches_control_limits_far_more_than_a_stable_population():
    """The whole point of anchoring limits to the baseline window: a drifting
    series must trip them, and a stationary one must mostly not."""
    stable = generate_population(n=24_000, seed=11, n_periods=12, drift_strength=0.0)
    drifting = generate_population(n=24_000, seed=11, n_periods=12, drift_strength=1.6)

    stable_breaches = stability_report(stable, 50_000, baseline_periods=3)["any_breach"].sum()
    drift_breaches = stability_report(drifting, 50_000, baseline_periods=3)["any_breach"].sum()

    assert stable_breaches <= 3
    assert drift_breaches >= 6
    assert drift_breaches > stable_breaches


def test_psi_by_period_baseline_is_zero_against_itself():
    population = generate_population(n=6_000, seed=3, n_periods=6)
    result = psi_by_period(population, "monthly_wire_value")
    assert result.iloc[0]["psi"] == 0.0
