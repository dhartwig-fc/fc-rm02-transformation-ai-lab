"""Report rendering and the synthetic generators."""

import numpy as np
import pandas as pd
import pytest

from tmtuning.data import generate_population, spec_population, split_by_period
from tmtuning.reporting import markdown_table, tuning_paper
from tmtuning.thresholds import optimise_threshold, threshold_sweep


# --- data ----------------------------------------------------------------

def test_spec_population_reproduces_the_spec_shape():
    df = spec_population(n=5_000, seed=42)
    assert len(df) == 5_000
    assert {"amount", "case"} <= set(df.columns)
    assert 0.02 < df["case"].mean() < 0.06


def test_spec_population_labels_are_independent_of_amount():
    """The Week 2 teaching point: precision is flat because there is no signal."""
    df = spec_population(n=40_000, seed=1)
    base_rate = df["case"].mean()
    for threshold in (25_000, 50_000, 75_000):
        alerted = df[df["amount"] > threshold]
        assert alerted["case"].mean() == pytest.approx(base_rate, abs=0.02)


def test_generate_population_hits_its_target_prevalence():
    for target in (0.01, 0.03, 0.05):
        df = generate_population(n=20_000, seed=5, prevalence=target)
        assert df["case"].mean() == pytest.approx(target, abs=0.006)


def test_generate_population_links_risk_to_value():
    """Everything from Week 3 onwards depends on precision actually rising."""
    df = generate_population(n=40_000, seed=5)
    precisions = [df.loc[df["monthly_wire_value"] > t, "case"].mean()
                  for t in (10_000, 25_000, 50_000, 100_000)]
    assert precisions == sorted(precisions)
    assert precisions[-1] > precisions[0] * 1.3


def test_generate_population_is_reproducible():
    a = generate_population(n=2_000, seed=99)
    b = generate_population(n=2_000, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_drift_increases_values_over_time():
    df = generate_population(n=30_000, seed=11, n_periods=12, drift_strength=1.6)
    first = df.loc[df["period_index"] == 0, "monthly_wire_value"].median()
    last = df.loc[df["period_index"] == 11, "monthly_wire_value"].median()
    assert last > first * 1.2


def test_no_drift_keeps_values_flat():
    df = generate_population(n=30_000, seed=11, n_periods=12, drift_strength=0.0)
    first = df.loc[df["period_index"] == 0, "monthly_wire_value"].median()
    last = df.loc[df["period_index"] == 11, "monthly_wire_value"].median()
    assert 0.8 < last / first < 1.25


@pytest.mark.parametrize("prevalence", [0.0, 1.0, -0.1])
def test_generate_population_rejects_impossible_prevalence(prevalence):
    with pytest.raises(ValueError, match="prevalence"):
        generate_population(n=100, prevalence=prevalence)


def test_split_by_period_is_disjoint_and_ordered():
    df = generate_population(n=12_000, seed=7, n_periods=12)
    in_time, out_of_time = split_by_period(df, holdout_periods=3)
    assert len(in_time) + len(out_of_time) == len(df)
    assert in_time["period_index"].max() < out_of_time["period_index"].min()
    assert out_of_time["period_index"].nunique() == 3


def test_split_by_period_refuses_to_consume_everything():
    df = generate_population(n=3_000, seed=7, n_periods=3)
    with pytest.raises(ValueError, match="no in-time data"):
        split_by_period(df, holdout_periods=3)


# --- reporting -----------------------------------------------------------

def test_markdown_table_keeps_integers_integral():
    """Regression: iterrows() upcasts a mixed-dtype row to one common dtype,
    which rendered an alert count of 10000 as '10,000.0'."""
    frame = pd.DataFrame({"alerts": [10_000, 250], "precision": [0.0448, 0.09]})
    rendered = markdown_table(frame, formats={"alerts": ",", "precision": ".2%"})
    assert "| 10,000 | 4.48% |" in rendered
    assert "10,000.0" not in rendered


def test_markdown_table_renders_nan_as_not_available():
    frame = pd.DataFrame({"precision": [float("nan")]})
    assert "n/a" in markdown_table(frame, formats={"precision": ".2%"})


def test_markdown_table_rejects_unknown_columns():
    with pytest.raises(KeyError, match="not in frame"):
        markdown_table(pd.DataFrame({"a": [1]}), columns=["b"])


def test_markdown_table_applies_header_overrides():
    frame = pd.DataFrame({"threshold": [1000]})
    assert "| Threshold (£) |" in markdown_table(frame, headers={"threshold": "Threshold (£)"})


@pytest.fixture(scope="module")
def paper_inputs():
    population = generate_population(n=10_000, seed=909, n_periods=12)
    sweep = threshold_sweep(population, "monthly_wire_value", "case", n_thresholds=20)
    return population, sweep, optimise_threshold(sweep, max_alerts=500)


def test_tuning_paper_contains_every_mandatory_section(paper_inputs):
    population, sweep, proposed = paper_inputs
    paper = tuning_paper("TM-014", "ALERT IF wire > t", population, sweep, proposed)
    for heading in ["Recommendation", "Rule under review", "Data and sample",
                    "Methodology", "Threshold sweep", "Limitations", "Approval"]:
        assert heading in paper


def test_tuning_paper_records_omitted_evidence_as_a_limitation(paper_inputs):
    """Skipping work should cost a sentence in the document, not go unnoticed."""
    population, sweep, proposed = paper_inputs
    paper = tuning_paper("TM-014", "ALERT IF wire > t", population, sweep, proposed)
    assert "No below-the-line test was performed" in paper
    assert "Segment-level calibration was not assessed" in paper


def test_tuning_paper_reports_deltas_in_percentage_points(paper_inputs):
    """A change from 7.53% to 7.87% is +0.34pp, not '+0.34%'."""
    population, sweep, proposed = paper_inputs
    current = sweep.iloc[0]
    paper = tuning_paper("TM-014", "ALERT IF wire > t", population, sweep, proposed, current)
    assert "pp)" in paper
