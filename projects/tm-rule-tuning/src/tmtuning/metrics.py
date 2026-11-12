"""Detection metrics for transaction monitoring rule tuning.

The vocabulary here is deliberately AML-first. A "positive prediction" is an
*alert*; a "positive label" is a *true case* (a customer/period the bank would,
with hindsight, want escalated). The metric names that matter to a tuning paper
are precision (alert yield), recall (detection rate) and the operational effort
ratio -- accuracy is computed but should never be quoted (see `accuracy`).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ConfusionCounts",
    "confusion_counts",
    "confusion_frame",
    "safe_divide",
    "classification_metrics",
    "metrics_frame",
]


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide, returning NaN rather than raising when the denominator is zero.

    Threshold sweeps routinely produce empty alert sets at the top of the range.
    A NaN propagates honestly through a results table; a 0.0 would be read as
    "this threshold has zero precision", which is a different -- and wrong --
    claim than "no alerts were generated, so precision is undefined".
    """
    denominator = float(denominator)
    if denominator == 0:
        return float("nan")
    return float(numerator) / denominator


def _as_bool_array(values: Iterable) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, (np.ndarray, pd.Series)) else values)
    if arr.dtype == object:
        arr = arr.astype(float)
    return arr.astype(bool)


@dataclass(frozen=True)
class ConfusionCounts:
    """The four cells of a detection confusion matrix.

    Attributes are named for what they mean operationally:

    * ``tp`` -- productive alerts (alerted and was a true case)
    * ``fp`` -- unproductive alerts (alerted, investigated, closed no-action)
    * ``fn`` -- missed risk (no alert, but was a true case)
    * ``tn`` -- correctly not alerted
    """

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def alerts(self) -> int:
        """Total alerts raised -- the number the operations team must work."""
        return self.tp + self.fp

    @property
    def cases(self) -> int:
        """Total true cases present in the population."""
        return self.tp + self.fn

    @property
    def population(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def as_dict(self) -> dict:
        return asdict(self)


def confusion_counts(y_true: Iterable, y_pred: Iterable) -> ConfusionCounts:
    """Count the four confusion cells from truth labels and alert flags.

    Both inputs are coerced to boolean, so 0/1 integers, True/False and
    "Yes"/"No" already mapped to 1/0 all work.

    >>> confusion_counts([1, 0, 1, 0], [1, 1, 0, 0])
    ConfusionCounts(tp=1, fp=1, fn=1, tn=1)
    """
    truth = _as_bool_array(y_true)
    pred = _as_bool_array(y_pred)
    if truth.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must be the same length, got {truth.shape} and {pred.shape}")

    return ConfusionCounts(
        tp=int(np.sum(truth & pred)),
        fp=int(np.sum(~truth & pred)),
        fn=int(np.sum(truth & ~pred)),
        tn=int(np.sum(~truth & ~pred)),
    )


def confusion_frame(y_true: Iterable, y_pred: Iterable) -> pd.DataFrame:
    """Confusion matrix as a labelled 2x2 frame, laid out for a tuning paper.

    Rows are the truth, columns are the rule's decision -- the orientation a
    reviewer expects, and the opposite of what ``print(sklearn_cm)`` shows
    without headers.
    """
    c = confusion_counts(y_true, y_pred)
    return pd.DataFrame(
        [[c.tp, c.fn], [c.fp, c.tn]],
        index=pd.Index(["True case", "Not a case"], name="Truth"),
        columns=pd.Index(["Alert", "No alert"], name="Rule decision"),
    )


def classification_metrics(y_true: Iterable, y_pred: Iterable) -> dict:
    """Full metric set for one rule configuration.

    Returns the confusion counts plus:

    ``precision``
        Productive alert rate (alert yield). Of the alerts we raise, what share
        are true cases? This is what drives investigator trust and cost.
    ``recall``
        Detection rate (coverage). Of the true cases present, what share did we
        alert on? This is what drives regulatory exposure.
    ``fpr``
        False positive rate: unproductive alerts as a share of all non-cases.
    ``f1``
        Harmonic mean of precision and recall -- a single tie-breaker, never a
        target in its own right (it weights a missed SAR the same as a wasted
        investigator hour, which no bank actually believes).
    ``specificity``
        Share of non-cases correctly left alone.
    ``alert_rate``
        Alerts as a share of the population -- the operational volume driver.
    ``prevalence``
        True cases as a share of the population -- the base rate.
    ``alerts_per_true_positive``
        Investigator effort ratio: how many alerts must be worked to surface one
        true case. The number an operations lead actually budgets against.
    ``accuracy``
        Computed for completeness only. At AML base rates (typically well under
        5%) a rule that alerts on nothing scores above 95% accuracy while
        detecting zero risk. Never quote it.
    """
    c = confusion_counts(y_true, y_pred)

    precision = safe_divide(c.tp, c.alerts)
    recall = safe_divide(c.tp, c.cases)

    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "tp": c.tp,
        "fp": c.fp,
        "fn": c.fn,
        "tn": c.tn,
        "alerts": c.alerts,
        "cases": c.cases,
        "population": c.population,
        "precision": precision,
        "recall": recall,
        "fpr": safe_divide(c.fp, c.fp + c.tn),
        "f1": f1,
        "specificity": safe_divide(c.tn, c.fp + c.tn),
        "alert_rate": safe_divide(c.alerts, c.population),
        "prevalence": safe_divide(c.cases, c.population),
        "alerts_per_true_positive": safe_divide(c.alerts, c.tp),
        "accuracy": safe_divide(c.tp + c.tn, c.population),
    }


def metrics_frame(results: Mapping[str, Mapping] | Sequence[Mapping]) -> pd.DataFrame:
    """Stack several metric dicts into a comparison table.

    Accepts either a mapping of ``{label: metrics_dict}`` (the label becomes the
    index) or a sequence of dicts that already carry their own identifying
    columns.
    """
    if isinstance(results, Mapping):
        frame = pd.DataFrame.from_dict(dict(results), orient="index")
        frame.index.name = "rule"
        return frame
    return pd.DataFrame(list(results))
