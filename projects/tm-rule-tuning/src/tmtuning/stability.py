"""Population stability, performance drift and control charts.

A threshold is tuned against one snapshot of behaviour. Behaviour then moves --
inflation, a new product, a portfolio acquisition, a seasonal pattern, a
typology shift -- and a threshold that was defensible in January quietly stops
being defensible by September. This module detects that before a regulator or
an alert backlog does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import classification_metrics
from .thresholds import apply_rule

__all__ = [
    "PSI_BANDS",
    "psi",
    "psi_by_period",
    "period_performance",
    "control_limits",
    "stability_report",
]

#: Conventional PSI interpretation bands, widely used in model risk practice.
PSI_BANDS = {
    "stable": (0.00, 0.10),
    "moderate shift": (0.10, 0.25),
    "significant shift": (0.25, float("inf")),
}

_EPS = 1e-6


def psi(expected: np.ndarray | pd.Series, actual: np.ndarray | pd.Series, bins: int = 10) -> float:
    """Population Stability Index between a baseline and a comparison sample.

    PSI = sum over bins of ``(actual% - expected%) * ln(actual% / expected%)``.

    Bin edges come from the *expected* (baseline) distribution's quantiles, so
    the baseline is roughly uniform across bins by construction and any
    departure is attributable to the actual sample.

    Interpretation (see :data:`PSI_BANDS`): below 0.10 is stable, 0.10-0.25 is a
    moderate shift worth explaining, above 0.25 is a significant shift that
    normally triggers re-tuning.

    Empty bins are floored at a small epsilon rather than dropped. Dropping them
    would silently understate the shift precisely when a whole region of the
    distribution has vanished -- which is the most important case to catch.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        raise ValueError("psi requires non-empty expected and actual samples")

    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError("expected sample has too little variation to form PSI bins")

    exp_pct = np.maximum(np.histogram(expected, bins=edges)[0] / expected.size, _EPS)
    act_pct = np.maximum(np.histogram(actual, bins=edges)[0] / actual.size, _EPS)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_band(value: float) -> str:
    """Label a PSI value with its conventional interpretation band."""
    for name, (lo, hi) in PSI_BANDS.items():
        if lo <= value < hi:
            return name
    return "significant shift"


def psi_by_period(
    df: pd.DataFrame,
    value_col: str,
    period_col: str = "period",
    baseline_period: str | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """PSI of ``value_col`` in every period against a fixed baseline period.

    The baseline defaults to the earliest period -- normally the data the
    threshold was tuned on, which is the comparison that matters.
    """
    periods = sorted(df[period_col].unique())
    baseline_period = periods[0] if baseline_period is None else baseline_period
    baseline = df.loc[df[period_col] == baseline_period, value_col]

    rows = []
    for p in periods:
        sample = df.loc[df[period_col] == p, value_col]
        value = 0.0 if p == baseline_period else psi(baseline, sample, bins=bins)
        rows.append({period_col: p, "n": int(sample.size), "psi": value, "band": psi_band(value)})
    return pd.DataFrame(rows)


def period_performance(
    df: pd.DataFrame,
    threshold: float,
    score_col: str = "monthly_wire_value",
    label_col: str = "case",
    period_col: str = "period",
    comparison: str = "gt",
) -> pd.DataFrame:
    """Rule metrics recomputed independently in each period.

    A single pooled backtest number hides everything interesting. Pooled
    precision of 8% is consistent with a rule that runs at a steady 8% and with
    one that ran at 14% in January and 3% by December -- and only one of those
    is a rule you can leave in production.
    """
    rows = []
    for p in sorted(df[period_col].unique()):
        part = df[df[period_col] == p]
        metrics = classification_metrics(part[label_col], apply_rule(part[score_col], threshold, comparison))
        rows.append({period_col: p, **metrics})
    return pd.DataFrame(rows)


def control_limits(series: pd.Series | np.ndarray, k: float = 3.0, method: str = "normal") -> dict:
    """Centre line and control limits for a monitoring metric.

    Parameters
    ----------
    method:
        ``"normal"`` uses mean +/- k standard deviations -- appropriate for rates
        and proportions such as precision or alert rate.

        ``"poisson"`` uses mean +/- k*sqrt(mean) -- appropriate for *counts*
        such as monthly alert volume, where the variance is tied to the level.
        Using normal limits on counts produces limits that are too wide at low
        volumes and too narrow at high ones.

    Lower limits are floored at zero: a negative alert count or precision is not
    a meaningful boundary.
    """
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("control_limits requires at least one non-NaN observation")

    centre = float(values.mean())
    if method == "normal":
        spread = float(values.std(ddof=1)) if values.size > 1 else 0.0
    elif method == "poisson":
        spread = float(np.sqrt(max(centre, 0.0)))
    else:
        raise ValueError(f"method must be 'normal' or 'poisson', got {method!r}")

    return {
        "centre": centre,
        "lower": max(centre - k * spread, 0.0),
        "upper": centre + k * spread,
        "k": k,
        "method": method,
    }


def stability_report(
    df: pd.DataFrame,
    threshold: float,
    score_col: str = "monthly_wire_value",
    label_col: str = "case",
    period_col: str = "period",
    k: float = 2.0,
    baseline_periods: int = 3,
    comparison: str = "gt",
) -> pd.DataFrame:
    """Per-period performance with control-limit breach flags.

    Combines :func:`period_performance`, :func:`psi_by_period` and
    :func:`control_limits` into the one table a monitoring pack needs.

    ``baseline_periods`` sets how many of the earliest periods define the
    control limits -- normally the window the threshold was tuned on. This is
    not a detail. Deriving limits from the *whole* series lets a steadily
    drifting metric inflate its own mean and standard deviation until the limits
    chase the drift and nothing ever breaches. Limits must be anchored to the
    period where the rule was known to be performing acceptably.

    ``k`` defaults to 2 rather than the manufacturing convention of 3. With only
    a handful of baseline months, 3-sigma limits on this much noise will
    essentially never trigger, and a control chart that cannot trigger provides
    assurance it has not earned.
    """
    if baseline_periods < 1:
        raise ValueError(f"baseline_periods must be at least 1, got {baseline_periods}")

    perf = period_performance(df, threshold, score_col, label_col, period_col, comparison)
    psi_frame = psi_by_period(df, score_col, period_col)
    report = perf.merge(psi_frame[[period_col, "psi", "band"]], on=period_col, how="left")

    if baseline_periods >= len(report):
        raise ValueError(
            f"baseline_periods={baseline_periods} covers the whole series "
            f"({len(report)} periods); leave at least one period to monitor."
        )
    baseline = report.head(baseline_periods)
    report["is_baseline"] = report.index < baseline_periods

    volume_limits = control_limits(baseline["alerts"], k=k, method="poisson")
    precision_limits = control_limits(baseline["precision"], k=k, method="normal")

    report["alert_volume_breach"] = (report["alerts"] < volume_limits["lower"]) | (report["alerts"] > volume_limits["upper"])
    report["precision_breach"] = (report["precision"] < precision_limits["lower"]) | (report["precision"] > precision_limits["upper"])
    report["psi_breach"] = report["psi"] >= 0.25
    report["any_breach"] = report[["alert_volume_breach", "precision_breach", "psi_breach"]].any(axis=1)

    report.attrs["volume_limits"] = volume_limits
    report.attrs["precision_limits"] = precision_limits
    return report
