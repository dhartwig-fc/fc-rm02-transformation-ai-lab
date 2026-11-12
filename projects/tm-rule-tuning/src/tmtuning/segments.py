"""Segment-level threshold calibration.

A single global threshold is a compromise between populations that do not
resemble each other. £50,000 a month is extraordinary for a retail customer and
unremarkable for a corporate; one number cannot be right for both, and the cost
of pretending otherwise is paid twice -- retail risk goes undetected while
corporate alerts flood the queue with normal business.

This module quantifies that cost and allocates a fixed alert budget across
segments to maximise total detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import safe_divide
from .thresholds import apply_rule, threshold_sweep

__all__ = [
    "concave_envelope",
    "segment_summary",
    "segment_sweeps",
    "uniform_threshold_at_capacity",
    "calibrate_segments",
    "compare_uniform_vs_segmented",
]


def segment_summary(
    df: pd.DataFrame,
    segment_col: str = "segment",
    label_col: str = "case",
    value_col: str = "monthly_wire_value",
) -> pd.DataFrame:
    """Profile each segment before tuning anything.

    Always run this first. It shows whether the segments differ enough in
    *value scale* and *base rate* to justify separate thresholds -- if they do
    not, segment-specific calibration adds governance overhead for no detection
    benefit, and you should say so rather than build it.
    """
    grouped = df.groupby(segment_col)
    out = pd.DataFrame({
        "rows": grouped.size(),
        "cases": grouped[label_col].sum(),
        "prevalence": grouped[label_col].mean(),
        "value_median": grouped[value_col].median(),
        "value_p90": grouped[value_col].quantile(0.90),
        "value_p99": grouped[value_col].quantile(0.99),
    })
    out["share_of_rows"] = out["rows"] / out["rows"].sum()
    out["share_of_cases"] = out["cases"] / out["cases"].sum()
    # >1 means the segment carries more risk than its size implies.
    out["case_concentration"] = out["share_of_cases"] / out["share_of_rows"]
    return out.sort_values("case_concentration", ascending=False)


def segment_sweeps(
    df: pd.DataFrame,
    score_col: str = "monthly_wire_value",
    label_col: str = "case",
    segment_col: str = "segment",
    n_thresholds: int = 40,
    comparison: str = "gt",
) -> dict[str, pd.DataFrame]:
    """One threshold sweep per segment, each on its *own* quantile grid.

    Using each segment's own distribution to set the candidate grid is
    essential. A grid derived from the pooled population is dominated by the
    largest segment, and would offer a corporate sweep almost no candidates in
    the range where its decision actually lies.
    """
    sweeps: dict[str, pd.DataFrame] = {}
    for name, part in df.groupby(segment_col):
        if part[label_col].sum() == 0:
            # No cases means no precision or recall to optimise. Skip rather
            # than emit a sweep of NaNs that looks like a result.
            continue
        sweeps[str(name)] = threshold_sweep(
            part, score_col=score_col, label_col=label_col,
            n_thresholds=n_thresholds, comparison=comparison,
        )
    return sweeps


def uniform_threshold_at_capacity(
    df: pd.DataFrame,
    capacity: int,
    score_col: str = "monthly_wire_value",
    label_col: str = "case",
    segment_col: str = "segment",
    n_thresholds: int = 200,
    comparison: str = "gt",
) -> pd.DataFrame:
    """The honest baseline: one global threshold tuned to the same alert budget.

    Any claim that segmentation helps must be made against a global threshold
    spending the *same* alert budget. Comparing a segmented rule against an
    untuned global threshold that happens to fire more alerts is not a
    comparison, it is a rigged one.
    """
    sweep = threshold_sweep(df, score_col, label_col, n_thresholds=n_thresholds, comparison=comparison)
    feasible = sweep[sweep["alerts"] <= capacity]
    if feasible.empty:
        raise ValueError(f"No global threshold fits a capacity of {capacity} alerts.")
    chosen = feasible.sort_values("alerts", ascending=False).iloc[0]
    threshold = float(chosen["threshold"])

    flags = apply_rule(df[score_col], threshold, comparison)
    rows = []
    for name, part in df.assign(_alert=flags).groupby(segment_col):
        tp = int(part.loc[part["_alert"], label_col].sum())
        alerts = int(part["_alert"].sum())
        rows.append({
            "segment": str(name), "threshold": threshold, "alerts": alerts, "tp": tp,
            "segment_cases": int(part[label_col].sum()),
            "precision": safe_divide(tp, alerts),
            "recall": safe_divide(tp, part[label_col].sum()),
        })
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def concave_envelope(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Upper concave envelope of ``(alerts, cases, threshold)`` points.

    Returns the subset of points lying on the upper-left convex hull, which is
    the sequence whose successive slopes strictly decrease.

    Why this is needed. A raw threshold grid produces a *noisy* marginal yield:
    one step of 237 alerts catches 5 cases, the next catches 2, the next 7. Fed
    to a greedy allocator that ranks steps by marginal yield, the noise decides
    the allocation -- a lucky step in a low-value segment outbids a genuinely
    productive run in a high-value one, and the result can be worse than a
    single global threshold.

    Taking the envelope first collapses any run of steps into one step carrying
    the run's *aggregate* slope. That removes the noise and, because the
    resulting slopes are strictly decreasing by construction, makes the greedy
    allocation in :func:`calibrate_segments` exactly optimal rather than merely
    near-optimal.
    """
    ordered = sorted(points, key=lambda p: (p[0], -p[1]))

    hull: list[tuple[float, float, float]] = []
    for point in ordered:
        if hull and hull[-1][0] == point[0]:
            if point[1] <= hull[-1][1]:
                continue
            hull.pop()
        while len(hull) >= 2:
            (a1, t1, _), (a2, t2, _) = hull[-2], hull[-1]
            # Drop the middle point when it sits on or below the chord a1 -> point,
            # i.e. when slope(a1, a2) <= slope(a2, point).
            if (t2 - t1) * (point[0] - a2) <= (point[1] - t2) * (a2 - a1):
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def calibrate_segments(
    df: pd.DataFrame,
    total_capacity: int,
    score_col: str = "monthly_wire_value",
    label_col: str = "case",
    segment_col: str = "segment",
    n_thresholds: int = 60,
    comparison: str = "gt",
) -> pd.DataFrame:
    """Allocate a total alert budget across segments to maximise detection.

    Algorithm
    ---------
    1. Sweep each segment on its own quantile grid.
    2. Reduce each segment's ``(alerts, cases)`` curve to its upper concave
       envelope (:func:`concave_envelope`), so marginal yield falls
       monotonically and sampling noise cannot drive the allocation.
    3. Greedily take whichever remaining step has the highest marginal yield and
       still fits the budget, until none does.

    In plain terms: spend each next alert wherever that alert is most likely to
    be productive -- the same logic a triage manager applies by instinct, made
    explicit and auditable.

    Because step 2 guarantees strictly decreasing marginal yield within each
    segment, the greedy pass in step 3 returns the optimal integral allocation
    over the candidate thresholds, up to the final partial step that the
    remaining budget cannot cover.

    Returns
    -------
    One row per segment with its calibrated threshold and resulting metrics,
    plus a ``TOTAL`` row.
    """
    if total_capacity < 0:
        raise ValueError(f"total_capacity must be non-negative, got {total_capacity}")

    sweeps = segment_sweeps(df, score_col, label_col, segment_col, n_thresholds, comparison)
    if not sweeps:
        raise ValueError("No segment contains any labelled cases; nothing to calibrate.")

    # Each ladder starts at "threshold = infinity, zero alerts, zero cases".
    ladders: dict[str, list[tuple[float, float, float]]] = {}
    raw_points: dict[str, list[tuple[float, float, float]]] = {}
    for name, sweep in sweeps.items():
        points = [(0.0, 0.0, float("inf"))]
        points += [(float(r.alerts), float(r.tp), float(r.threshold)) for r in sweep.itertuples()]
        raw_points[name] = sorted(points, key=lambda p: p[0])
        ladders[name] = concave_envelope(points)

    # --- Lagrangian allocation -------------------------------------------
    # For a shadow price `lam` (cases per alert), each segment independently
    # picks the envelope point maximising `cases - lam * alerts`. Because the
    # envelopes are concave, that per-segment choice is exactly optimal, and
    # total alerts fall monotonically as `lam` rises -- so bisecting `lam` to
    # meet the budget solves the whole allocation at once.
    #
    # This replaces a straight best-first greedy pass, which suffers knapsack
    # fragmentation: once the remaining budget is small, greedy takes a cheap
    # low-yield step in one segment simply because the genuinely better step
    # elsewhere no longer fits, and the result can be worse than a single
    # global threshold.
    def _choose(lam: float) -> dict[str, int]:
        chosen = {}
        for name, ladder in ladders.items():
            values = [cases - lam * alerts for alerts, cases, _ in ladder]
            best = max(range(len(ladder)), key=lambda i: (values[i], -ladder[i][0]))
            chosen[name] = best
        return chosen

    def _total_alerts(chosen: dict[str, int]) -> int:
        return int(sum(ladders[n][i][0] for n, i in chosen.items()))

    lo, hi = 0.0, 1.0  # a slope of 1.0 means "every extra alert is a case"
    if _total_alerts(_choose(lo)) <= total_capacity:
        position = _choose(lo)
    else:
        for _ in range(60):
            mid = (lo + hi) / 2
            if _total_alerts(_choose(mid)) > total_capacity:
                lo = mid
            else:
                hi = mid
        position = _choose(hi)

    # Top-up: bisection lands on the largest allocation within budget, which can
    # leave a remainder too small for any segment's next *envelope* step -- those
    # steps get coarse at the loose end, where one vertex can span thousands of
    # alerts. So the remainder is spent against the full grid instead, which
    # offers finer moves. The envelope decides the allocation; the raw grid just
    # uses up the change.
    state = {name: ladders[name][i] for name, i in position.items()}
    spent = int(sum(alerts for alerts, _, _ in state.values()))

    while True:
        best_name, best_gain, best_point = None, 0.0, None
        for name, points in raw_points.items():
            cur_alerts, cur_tp, _ = state[name]
            for alerts, tp, threshold in points:
                extra_alerts = int(alerts - cur_alerts)
                if extra_alerts <= 0 or spent + extra_alerts > total_capacity:
                    continue
                gain = (tp - cur_tp) / extra_alerts
                if gain > best_gain:
                    best_name, best_gain, best_point = name, gain, (alerts, tp, threshold)
        if best_name is None:
            break
        spent += int(best_point[0] - state[best_name][0])
        state[best_name] = best_point

    case_totals = df.groupby(segment_col)[label_col].sum()

    rows = []
    for name, (alerts, tp, threshold) in state.items():
        rows.append({
            "segment": name, "threshold": float(threshold), "alerts": int(alerts), "tp": int(tp),
            "segment_cases": int(case_totals.get(name, 0)),
            "precision": safe_divide(tp, alerts),
            "recall": safe_divide(tp, case_totals.get(name, 0)),
        })

    result = pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)
    total = {
        "segment": "TOTAL", "threshold": np.nan,
        "alerts": int(result["alerts"].sum()), "tp": int(result["tp"].sum()),
        "segment_cases": int(df[label_col].sum()),
        "precision": safe_divide(result["tp"].sum(), result["alerts"].sum()),
        "recall": safe_divide(result["tp"].sum(), df[label_col].sum()),
    }
    return pd.concat([result, pd.DataFrame([total])], ignore_index=True)


def compare_uniform_vs_segmented(
    df: pd.DataFrame,
    capacity: int,
    score_col: str = "monthly_wire_value",
    label_col: str = "case",
    segment_col: str = "segment",
    n_thresholds: int = 60,
    comparison: str = "gt",
) -> pd.DataFrame:
    """Side-by-side of one global threshold vs per-segment thresholds, same budget.

    The ``uplift_tp`` and ``uplift_recall`` columns on the ``TOTAL`` row are the
    headline finding: extra true cases detected for no extra investigator
    effort. If that uplift is small, segmentation is not worth its governance
    cost, and the paper should recommend against it.
    """
    uniform = uniform_threshold_at_capacity(
        df, capacity, score_col, label_col, segment_col,
        n_thresholds=max(n_thresholds, 200), comparison=comparison,
    )
    uniform_total = pd.DataFrame([{
        "segment": "TOTAL", "threshold": uniform["threshold"].iloc[0],
        "alerts": int(uniform["alerts"].sum()), "tp": int(uniform["tp"].sum()),
        "segment_cases": int(df[label_col].sum()),
        "precision": safe_divide(uniform["tp"].sum(), uniform["alerts"].sum()),
        "recall": safe_divide(uniform["tp"].sum(), df[label_col].sum()),
    }])
    uniform = pd.concat([uniform, uniform_total], ignore_index=True)

    segmented = calibrate_segments(
        df, capacity, score_col, label_col, segment_col, n_thresholds, comparison
    )

    merged = uniform.merge(segmented, on="segment", suffixes=("_uniform", "_segmented"))
    merged["uplift_tp"] = merged["tp_segmented"] - merged["tp_uniform"]
    merged["uplift_recall"] = merged["recall_segmented"] - merged["recall_uniform"]
    merged["alert_delta"] = merged["alerts_segmented"] - merged["alerts_uniform"]
    return merged
