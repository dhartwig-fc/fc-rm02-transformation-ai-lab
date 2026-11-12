"""Champion/challenger comparison and ATL/BTL statistics."""

import numpy as np
import pytest

from tmtuning.challenger import (atl_btl_sample, btl_test, compare_rules, required_sample_size,
                                 rule_overlap, wilson_interval)
from tmtuning.data import generate_population
from tmtuning.thresholds import apply_rule


@pytest.fixture(scope="module")
def population():
    return generate_population(n=20_000, seed=808, n_periods=12)


def test_wilson_interval_matches_published_values():
    """0 successes in 100 trials: the textbook Wilson upper bound is 0.0370."""
    low, high = wilson_interval(0, 100, 0.95)
    assert low == 0.0
    assert high == pytest.approx(0.0370, abs=5e-4)


def test_wilson_interval_is_symmetric_at_one_half():
    low, high = wilson_interval(50, 100, 0.95)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)
    assert (low + high) / 2 == pytest.approx(0.5)


def test_wilson_interval_has_width_at_zero_successes():
    """The reason Wilson is used instead of Wald: Wald returns [0, 0] here, which
    would claim with certainty that nothing sits below the line."""
    assert wilson_interval(0, 500)[1] > 0


def test_wilson_interval_narrows_as_the_sample_grows():
    widths = [wilson_interval(int(0.02 * n), n)[1] - wilson_interval(int(0.02 * n), n)[0]
              for n in (100, 1_000, 10_000)]
    assert widths == sorted(widths, reverse=True)


def test_higher_confidence_gives_a_wider_interval():
    narrow = wilson_interval(20, 500, 0.90)
    wide = wilson_interval(20, 500, 0.99)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


@pytest.mark.parametrize("successes,trials", [(-1, 10), (11, 10), (5, 0)])
def test_wilson_interval_rejects_impossible_inputs(successes, trials):
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)


def test_required_sample_size_matches_the_standard_formula():
    # n = z^2 p(1-p) / e^2  with z = 1.959964, p = 0.03, e = 0.01
    assert required_sample_size(0.03, 0.01, 0.95) == 1118


def test_required_sample_size_quadruples_when_margin_halves():
    assert required_sample_size(0.02, 0.005) == pytest.approx(
        required_sample_size(0.02, 0.01) * 4, rel=0.01)


def test_required_sample_size_uses_the_conservative_variance_at_the_extremes():
    assert required_sample_size(0.0, 0.05) == required_sample_size(0.5, 0.05)


def test_compare_rules_adds_deltas_against_the_champion(population):
    wires = population["monthly_wire_value"]
    result = compare_rules(population,
                           {"champ": apply_rule(wires, 50_000), "chall": apply_rule(wires, 75_000)},
                           champion="champ")
    assert result.loc["champ", "delta_tp"] == 0
    assert result.loc["chall", "delta_alerts"] < 0


def test_compare_rules_rejects_a_missing_champion(population):
    with pytest.raises(KeyError, match="not among the rules"):
        compare_rules(population, {"a": np.ones(len(population), dtype=bool)}, champion="b")


def test_compare_rules_rejects_misaligned_flags(population):
    with pytest.raises(ValueError, match="flags for"):
        compare_rules(population, {"a": np.ones(5, dtype=bool)})


def test_rule_overlap_cells_partition_the_population(population):
    wires = population["monthly_wire_value"]
    overlap = rule_overlap(population, apply_rule(wires, 50_000), apply_rule(wires, 75_000))
    assert overlap["rows"].sum() == len(population)
    assert overlap["cases"].sum() == population["case"].sum()
    assert overlap["share_of_population"].sum() == pytest.approx(1.0)


def test_rule_overlap_exposes_netting(population):
    """A tightened rule can only lose cases, never gain them."""
    wires = population["monthly_wire_value"]
    overlap = rule_overlap(population, apply_rule(wires, 50_000), apply_rule(wires, 75_000))
    assert overlap.loc[overlap["cell"] == "Challenger only", "rows"].iloc[0] == 0
    assert overlap.loc[overlap["cell"] == "Champion only", "rows"].iloc[0] > 0


def test_atl_btl_sample_draws_from_the_right_sides(population):
    flags = apply_rule(population["monthly_wire_value"], 50_000)
    samples = atl_btl_sample(population, flags, n_above=50, n_below=100, seed=1)
    assert len(samples["above"]) == 50
    assert len(samples["below"]) == 100
    assert (samples["above"]["monthly_wire_value"] > 50_000).all()
    assert (samples["below"]["monthly_wire_value"] <= 50_000).all()


def test_atl_btl_sample_is_reproducible(population):
    flags = apply_rule(population["monthly_wire_value"], 50_000)
    first = atl_btl_sample(population, flags, 20, 40, seed=7)["below"]
    second = atl_btl_sample(population, flags, 20, 40, seed=7)["below"]
    assert first["customer_id"].tolist() == second["customer_id"].tolist()


def test_atl_btl_sample_caps_at_the_pool_size(population):
    flags = apply_rule(population["monthly_wire_value"], 50_000)
    samples = atl_btl_sample(population, flags, n_above=10**9, n_below=10, seed=1)
    assert len(samples["above"]) == int(flags.sum())


def test_btl_test_interval_covers_the_true_rate(population):
    """With a 2,000-record sample the interval should contain the truth."""
    flags = apply_rule(population["monthly_wire_value"], 50_000)
    result = btl_test(population, flags, n_below=2_000, seed=808)
    true_rate = result["actual_missed_cases"] / result["below_the_line_population"]
    assert result["rate_lower"] <= true_rate <= result["rate_upper"]


def test_btl_upper_bound_exceeds_the_point_estimate(population):
    flags = apply_rule(population["monthly_wire_value"], 50_000)
    result = btl_test(population, flags, n_below=800, seed=808)
    assert result["estimated_missed_cases_upper"] > result["estimated_missed_cases"]


def test_btl_test_errors_when_everything_alerts(population):
    with pytest.raises(ValueError, match="no below-the-line"):
        btl_test(population, np.ones(len(population), dtype=bool))
