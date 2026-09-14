"""Threshold sweeps and capacity-constrained threshold selection.

The central object is the *sweep*: one row per candidate threshold, carrying the
full metric set at that threshold. Everything else in this module either builds
a sweep, reads one, or picks a row out of one.
"""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .metrics import classification_metrics, safe_divide

__all__ = [
    "Comparison",
    "apply_rule",
    "default_thresholds",
    "threshold_sweep",
    "marginal_yield",
    "optimise_threshold",
    "capacity_frontier",
]

Comparison = Literal["gt", "ge"]


def apply_rule(values: pd.Series | np.ndarray, threshold: float, comparison: Comparison = "gt") -> np.ndarray:
    """Return the boolean alert flag for ``values`` against ``threshold``.

    ``comparison`` matters more than it looks. A rule written "alert if monthly
    cash deposits > £10,000" (``gt``) does **not** alert on a deposit of exactly
    £10,000 -- and depositing exactly the round-number threshold is textbook
    structuring behaviour. On smooth synthetic amounts the choice is a
    zero-measure difference; on real data, where round numbers pile up at
    exactly the threshold, it can move the alert count materially. Check which
    one the production engine implements before you quote a tuning result.
    """
    arr = np.asarray(values, dtype=float)
    if comparison == "gt":
        return arr > threshold
    if comparison == "ge":
        return arr >= threshold
    raise ValueError(f"comparison must be 'gt' or 'ge', got {comparison!r}")


def default_thresholds(
    values: pd.Series | np.ndarray,
    n_thresholds: int = 40,
    q_low: float = 0.05,
    q_high: float = 0.999,
) -> np.ndarray:
    """Build a candidate threshold grid from the data's own quantiles.

    Quantile spacing, not linear spacing. Transaction values are heavily
    right-skewed, so a linear grid over the range spends most of its candidates
    in a tail that contains almost no customers, and leaves the dense region
    where the threshold decision actually lives almost unsampled.

    The range runs from the 5th to the 99.9th percentile by default, and half
    the candidates are concentrated above the 95th. The low end matters more
    than it looks: a grid starting at the median caps any rule at alerting on
    50% of its population, which silently makes loose thresholds unreachable.
    That is invisible on a pooled sweep but breaks segment calibration outright,
    because a high-value segment may need a threshold below its own median to
    match the alert volume a global threshold gives it.
    """
    if n_thresholds < 2:
        raise ValueError(f"n_thresholds must be at least 2, got {n_thresholds}")
    if not 0 <= q_low < q_high <= 1:
        raise ValueError(f"require 0 <= q_low < q_high <= 1, got {q_low} and {q_high}")

    arr = np.asarray(values, dtype=float)
    split = max(n_thresholds // 2, 1)
    q_mid = min(0.95, (q_low + q_high) / 2)
    quantiles = np.concatenate([
        np.linspace(q_low, q_mid, split, endpoint=False),
        np.linspace(q_mid, q_high, n_thresholds - split),
    ])
    return np.unique(np.quantile(arr, quantiles).round(2))


def threshold_sweep(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "case",
    thresholds: Iterable[float] | None = None,
    n_thresholds: int = 40,
    comparison: Comparison = "gt",
) -> pd.DataFrame:
    """Evaluate a rule at every candidate threshold.

    Parameters
    ----------
    df:
        Population to score. One row per scored unit (typically customer-month).
    score_col:
        The column the rule thresholds on, e.g. ``monthly_wire_value``.
    label_col:
        Binary outcome column: 1 for a true case.
    thresholds:
        Explicit candidates. When omitted, a quantile grid from
        :func:`default_thresholds` is used.
    n_thresholds:
        Grid size when ``thresholds`` is omitted.
    comparison:
        ``"gt"`` (default, matching rules written "> £X") or ``"ge"``.

    Returns
    -------
    One row per threshold, sorted ascending, with the columns produced by
    :func:`~tmtuning.metrics.classification_metrics` plus ``threshold``.
    """
    for col in (score_col, label_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not found; available: {list(df.columns)}")

    candidates = (
        default_thresholds(df[score_col], n_thresholds) if thresholds is None else np.sort(np.asarray(list(thresholds), dtype=float))
    )

    labels = df[label_col].to_numpy()
    values = df[score_col].to_numpy()

    rows = []
    for t in candidates:
        metrics = classification_metrics(labels, apply_rule(values, t, comparison))
        rows.append({"threshold": float(t), **metrics})

    return pd.DataFrame(rows)


def marginal_yield(sweep: pd.DataFrame) -> pd.DataFrame:
    """Incremental productivity of *loosening* the threshold one step.

    Walks the sweep from the tightest threshold downwards. Each row answers: if
    I relax the rule from the threshold above to this one, how many extra alerts
    do I take on, how many extra true cases do I catch, and what is the yield on
    that increment?

    ``marginal_precision`` is the number that should drive a threshold decision,
    and it is almost always worse than the headline precision at that threshold.
    Cumulative precision averages in the highly productive top of the
    distribution; the marginal figure prices only the alerts you are actually
    about to add. A step whose marginal precision sits near the population base
    rate is buying alerts at random -- that is where the threshold belongs.
    """
    ordered = sweep.sort_values("threshold", ascending=False).reset_index(drop=True)
    extra_alerts = -ordered["alerts"].diff(-1).shift(1)
    extra_tp = -ordered["tp"].diff(-1).shift(1)

    out = ordered[["threshold", "alerts", "tp", "precision", "recall"]].copy()
    out["extra_alerts"] = extra_alerts
    out["extra_tp"] = extra_tp
    out["marginal_precision"] = [
        safe_divide(tp, al) for tp, al in zip(out["extra_tp"], out["extra_alerts"])
    ]
    out["alerts_per_extra_tp"] = [
        safe_divide(al, tp) for tp, al in zip(out["extra_tp"], out["extra_alerts"])
    ]
    return out


def optimise_threshold(
    sweep: pd.DataFrame,
    objective: str = "recall",
    max_alerts: int | None = None,
    max_alert_rate: float | None = None,
    min_precision: float | None = None,
    min_recall: float | None = None,
    strict: bool = True,
) -> pd.Series | None:
    """Pick the best threshold subject to operational constraints.

    This is the shape a real tuning decision takes. Operations can work a fixed
    number of alerts per month; within that budget, the rule should surface as
    much risk as possible. So the honest formulation is *constrained*:

        maximise detection, subject to alerts <= capacity

    not "maximise F1", which silently prices a missed case and a wasted
    investigator hour identically.

    Parameters
    ----------
    sweep:
        Output of :func:`threshold_sweep`.
    objective:
        Column to maximise -- ``"recall"`` (default), ``"tp"``, ``"precision"``
        or ``"f1"``.
    max_alerts, max_alert_rate:
        Operational capacity, as an absolute alert count or as a share of the
        population. Apply whichever the operating model is expressed in.
    min_precision, min_recall:
        Quality floors. A precision floor keeps investigators from drowning in
        noise; a recall floor is often set by risk appetite or by not regressing
        against the incumbent rule.
    strict:
        Raise when no threshold satisfies the constraints. Set ``False`` to get
        ``None`` back instead.

    Returns
    -------
    The winning sweep row, or ``None`` when infeasible and ``strict`` is False.
    """
    if objective not in sweep.columns:
        raise KeyError(f"objective {objective!r} is not a column of the sweep; available: {list(sweep.columns)}")

    feasible = sweep.copy()
    applied: list[str] = []

    if max_alerts is not None:
        feasible = feasible[feasible["alerts"] <= max_alerts]
        applied.append(f"alerts <= {max_alerts}")
    if max_alert_rate is not None:
        feasible = feasible[feasible["alert_rate"] <= max_alert_rate]
        applied.append(f"alert_rate <= {max_alert_rate:.4f}")
    if min_precision is not None:
        feasible = feasible[feasible["precision"] >= min_precision]
        applied.append(f"precision >= {min_precision:.4f}")
    if min_recall is not None:
        feasible = feasible[feasible["recall"] >= min_recall]
        applied.append(f"recall >= {min_recall:.4f}")

    feasible = feasible[feasible[objective].notna()]

    if feasible.empty:
        if strict:
            raise ValueError(
                "No threshold satisfies the constraints "
                f"[{', '.join(applied) or 'none'}]. Widen the grid, relax a "
                "constraint, or accept that this rule cannot meet the target."
            )
        return None

    # Ties broken towards the *tighter* threshold: same detection for fewer
    # alerts is strictly better operationally.
    best = feasible.sort_values([objective, "threshold"], ascending=[False, False]).iloc[0]
    return best


def capacity_frontier(sweep: pd.DataFrame, capacities: Sequence[int], objective: str = "recall") -> pd.DataFrame:
    """Best achievable detection at each of several alert-capacity budgets.

    The table to put in front of an operations lead who is deciding how many
    investigators to fund: each row says what a given monthly alert budget buys
    in detection, and what threshold delivers it.
    """
    rows = []
    for capacity in capacities:
        best = optimise_threshold(sweep, objective=objective, max_alerts=capacity, strict=False)
        if best is None:
            rows.append({"capacity": capacity, "threshold": np.nan, "alerts": np.nan,
                         "tp": np.nan, "precision": np.nan, "recall": np.nan})
        else:
            rows.append({
                "capacity": capacity,
                "threshold": best["threshold"],
                "alerts": int(best["alerts"]),
                "tp": int(best["tp"]),
                "precision": best["precision"],
                "recall": best["recall"],
            })
    return pd.DataFrame(rows)
