"""Metric arithmetic, checked against hand-computed values."""

import math

import numpy as np
import pytest

from tmtuning.metrics import (classification_metrics, confusion_counts, confusion_frame,
                              metrics_frame, safe_divide)

# The Week 1 worked example: customers A/B/C/D -> TP, FP, FN, TN.
Y_TRUE = [1, 0, 1, 0]
Y_PRED = [1, 1, 0, 0]


def test_confusion_counts_matches_worked_example():
    counts = confusion_counts(Y_TRUE, Y_PRED)
    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (1, 1, 1, 1)
    assert counts.alerts == 2
    assert counts.cases == 2
    assert counts.population == 4


def test_confusion_counts_accepts_bools_and_arrays():
    assert confusion_counts([True, False], [True, True]) == confusion_counts([1, 0], [1, 1])
    assert confusion_counts(np.array([1, 0]), np.array([1, 1])).fp == 1


def test_confusion_counts_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        confusion_counts([1, 0, 1], [1, 0])


def test_confusion_frame_orientation():
    """Rows are truth, columns are the rule's decision -- not sklearn's layout."""
    frame = confusion_frame(Y_TRUE, Y_PRED)
    assert frame.loc["True case", "Alert"] == 1       # TP
    assert frame.loc["True case", "No alert"] == 1    # FN
    assert frame.loc["Not a case", "Alert"] == 1      # FP
    assert frame.loc["Not a case", "No alert"] == 1   # TN


@pytest.mark.parametrize("metric,expected", [
    ("precision", 0.5), ("recall", 0.5), ("fpr", 0.5), ("f1", 0.5),
    ("specificity", 0.5), ("accuracy", 0.5), ("alert_rate", 0.5),
    ("prevalence", 0.5), ("alerts_per_true_positive", 2.0),
])
def test_classification_metrics_values(metric, expected):
    assert classification_metrics(Y_TRUE, Y_PRED)[metric] == pytest.approx(expected)


def test_safe_divide_returns_nan_not_zero():
    """A zero denominator means 'undefined', which is not the same claim as zero."""
    assert math.isnan(safe_divide(1, 0))
    assert safe_divide(1, 2) == 0.5


def test_precision_is_nan_when_no_alerts_raised():
    metrics = classification_metrics([1, 0, 1, 0], [0, 0, 0, 0])
    assert math.isnan(metrics["precision"])
    assert metrics["recall"] == 0.0
    assert metrics["alerts"] == 0


def test_f1_is_nan_when_precision_undefined():
    assert math.isnan(classification_metrics([1, 0], [0, 0])["f1"])


def test_accuracy_trap_at_low_base_rate():
    """The Week 1 demonstration: alerting on nothing scores 97% accuracy."""
    truth = np.zeros(10_000, dtype=int)
    truth[:300] = 1  # 3% base rate
    metrics = classification_metrics(truth, np.zeros(10_000, dtype=int))
    assert metrics["accuracy"] == pytest.approx(0.97)
    assert metrics["recall"] == 0.0


def test_metrics_frame_from_mapping_and_sequence():
    mapping = metrics_frame({"a": classification_metrics(Y_TRUE, Y_PRED)})
    assert mapping.index.tolist() == ["a"]
    assert len(metrics_frame([classification_metrics(Y_TRUE, Y_PRED)])) == 1
