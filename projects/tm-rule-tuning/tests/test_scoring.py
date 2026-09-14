"""Weighted scoring, deciles and score cut-offs (Week 8)."""

import numpy as np
import pandas as pd
import pytest

from tmtuning.data import generate_population
from tmtuning.scoring import (band_score, calibrate_probability, decile_table, optimise_cutoff,
                              percentile_score, score_cutoff_sweep, weighted_score)

GEO_BANDS = {"DOMESTIC": 0, "STANDARD": 40, "ELEVATED": 75, "HIGH": 100}


@pytest.fixture(scope="module")
def scored():
    df = generate_population(n=20_000, seed=808, n_periods=12)
    df = df.assign(
        amount_score=percentile_score(df["monthly_wire_value"]),
        velocity_score=percentile_score(df["velocity"]),
        geo_score=band_score(df["jurisdiction_risk"], GEO_BANDS),
    )
    df["score"] = weighted_score({
        "amount": (df["amount_score"], 0.5),
        "velocity": (df["velocity_score"], 0.3),
        "geo": (df["geo_score"], 0.2),
    })
    return df


def test_percentile_score_spans_the_full_range():
    values = np.arange(100.0)
    scores = percentile_score(values)
    assert scores.min() == pytest.approx(1.0)
    assert scores.max() == pytest.approx(100.0)
    assert (np.diff(scores) > 0).all()


def test_percentile_score_averages_ties():
    """Discrete counts pile up on ties; without averaging the component would
    carry ordering the data does not support."""
    scores = percentile_score([0, 0, 0, 0, 5])
    assert len(set(scores[:4])) == 1
    assert scores[4] > scores[0]


def test_percentile_score_is_invariant_to_monotone_rescaling():
    base = np.array([1.0, 10.0, 100.0, 1000.0])
    assert percentile_score(base) == pytest.approx(percentile_score(base * 7.3))


def test_band_score_maps_and_defaults():
    scores = band_score(["HIGH", "DOMESTIC", "UNKNOWN"], GEO_BANDS, default=-1)
    assert scores.tolist() == [100.0, 0.0, -1.0]


def test_weighted_score_matches_manual_arithmetic():
    a, b = np.array([10.0, 20.0]), np.array([100.0, 0.0])
    result = weighted_score({"a": (a, 0.5), "b": (b, 0.5)})
    assert result.tolist() == [55.0, 10.0]


def test_weighted_score_rejects_weights_that_do_not_sum_to_one():
    """Weights summing to anything else silently rescale the score, so a fixed
    cut-off quietly means something different from one run to the next."""
    with pytest.raises(ValueError, match="sum to"):
        weighted_score({"a": (np.array([1.0]), 0.5), "b": (np.array([1.0]), 0.3)})


def test_weighted_score_allows_deliberate_override():
    result = weighted_score({"a": (np.array([10.0]), 0.5), "b": (np.array([10.0]), 0.3)},
                            strict_weights=False)
    assert result.tolist() == [8.0]


def test_weighted_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differing lengths"):
        weighted_score({"a": (np.array([1.0, 2.0]), 0.5), "b": (np.array([1.0]), 0.5)})


def test_weighted_score_rejects_empty_components():
    with pytest.raises(ValueError, match="at least one component"):
        weighted_score({})


def test_decile_table_partitions_the_population(scored):
    table = decile_table(scored, "score", "case")
    assert len(table) == 10
    assert table["records"].sum() == len(scored)
    assert table["cases"].sum() == scored["case"].sum()
    assert table["cumulative_recall"].iloc[-1] == pytest.approx(1.0)


def test_decile_one_holds_the_highest_scores(scored):
    table = decile_table(scored, "score", "case")
    assert table["min_score"].iloc[0] >= table["max_score"].iloc[1]


def test_score_separates_risk(scored):
    """The whole premise of Week 8: the top decile must carry materially more
    risk than the bottom, or the score is not worth its governance cost."""
    table = decile_table(scored, "score", "case")
    assert table["lift"].iloc[0] > 3.0
    assert table["case_rate"].iloc[0] > table["case_rate"].iloc[-1] * 5
    assert table["cumulative_recall"].iloc[0] > 0.3


def test_decile_table_rejects_a_missing_score_column(scored):
    with pytest.raises(KeyError, match="not found"):
        decile_table(scored, "no_such_column", "case")


def test_cutoff_sweep_alerts_fall_as_the_cutoff_rises(scored):
    sweep = score_cutoff_sweep(scored, range(20, 95, 5), "score", "case")
    assert sweep["alerts"].is_monotonic_decreasing
    assert sweep["recall"].is_monotonic_decreasing


def test_optimise_cutoff_respects_capacity(scored):
    sweep = score_cutoff_sweep(scored, range(20, 95, 5), "score", "case")
    best = optimise_cutoff(sweep, objective="recall", max_alerts=1_000)
    assert best["alerts"] <= 1_000


def test_optimise_cutoff_raises_when_infeasible(scored):
    sweep = score_cutoff_sweep(scored, range(20, 95, 5), "score", "case")
    with pytest.raises(ValueError, match="No score cut-off"):
        optimise_cutoff(sweep, max_alerts=1, min_precision=0.99)


def test_calibration_bands_carry_confidence_intervals(scored):
    calibration = calibrate_probability(scored, "score", "case", n_bins=8)
    assert (calibration["ci_low"] <= calibration["observed_probability"]).all()
    assert (calibration["observed_probability"] <= calibration["ci_high"]).all()
    assert calibration["records"].sum() == len(scored)


def test_score_beats_a_single_threshold_at_matched_volume(scored):
    """Week 8's headline claim, asserted rather than assumed."""
    from tmtuning.metrics import classification_metrics
    from tmtuning.thresholds import apply_rule

    incumbent = apply_rule(scored["monthly_wire_value"], 50_000)
    target = int(np.asarray(incumbent).sum())
    cut = float(np.quantile(scored["score"], 1 - target / len(scored)))

    by_threshold = classification_metrics(scored["case"], incumbent)
    by_score = classification_metrics(scored["case"], scored["score"] >= cut)

    assert abs(by_score["alerts"] - by_threshold["alerts"]) < target * 0.05
    assert by_score["tp"] > by_threshold["tp"]
