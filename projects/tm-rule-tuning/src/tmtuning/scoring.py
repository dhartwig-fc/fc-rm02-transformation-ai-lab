"""Weighted risk scoring and score-based calibration.

A single threshold forces one variable to carry the whole decision. A weighted
score keeps several, and keeps the gradient within each: a customer just under
the amount cut but well above on velocity and sitting in a high-risk
jurisdiction can still surface.

The scores built here are deliberately transparent -- percentile ranks combined
with weights a person chose and can defend. A fitted model would score better.
It would also be far harder to put in front of a validator, and the weights
would no longer trace to anything a reviewer could argue with, which is the
trade Week 8 asks you to make explicitly.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .metrics import classification_metrics, safe_divide

__all__ = [
    "percentile_score",
    "band_score",
    "weighted_score",
    "decile_table",
    "score_cutoff_sweep",
    "calibrate_probability",
    "optimise_cutoff",
]


def percentile_score(values: pd.Series | np.ndarray, scale: float = 100.0) -> np.ndarray:
    """Percentile rank of each value, scaled to 0-``scale``.

    Ties take their average rank, which matters for discrete inputs such as a
    transaction count: without it every customer with velocity 0 would be
    ranked arbitrarily among themselves, and the component score would carry
    ordering that the data does not support.

    Percentile ranks also make components *commensurable*. Raw pounds and raw
    transaction counts cannot be added; their ranks can, which is what lets a
    set of weights mean what it appears to mean.
    """
    series = pd.Series(np.asarray(values, dtype=float))
    return (series.rank(pct=True, method="average") * scale).to_numpy()


def band_score(values: pd.Series | np.ndarray, bands: Mapping, default: float = 0.0) -> np.ndarray:
    """Map categorical values to scores via an explicit lookup.

    Used for inputs where a rank makes no sense -- a jurisdiction tier is an
    ordered judgement, not a measurement, so its score should be a number
    somebody chose and wrote down.
    """
    series = pd.Series(np.asarray(values))
    return series.map(lambda v: bands.get(v, default)).astype(float).to_numpy()


def weighted_score(components: Mapping[str, tuple[np.ndarray, float]],
                   strict_weights: bool = True) -> np.ndarray:
    """Combine component scores under explicit weights.

    Parameters
    ----------
    components:
        ``{name: (component_score, weight)}``.
    strict_weights:
        Require the weights to sum to 1. On by default, because weights that do
        not sum to 1 silently rescale the score: a cut-off of 60 then means
        something different from one run to the next, and nobody notices until
        the alert volume moves.

    Every weight is a judgement about relative importance. State the rationale
    for each one in the tuning paper -- "amount is weighted 0.5 because ..." --
    since a validator will ask, and "it performed best" is circular when the
    weights were chosen on the same data the performance was measured on.
    """
    if not components:
        raise ValueError("weighted_score requires at least one component")

    total = sum(weight for _, weight in components.values())
    if strict_weights and not np.isclose(total, 1.0):
        raise ValueError(
            f"component weights sum to {total:.4f}, not 1.0. Either fix the weights "
            "or pass strict_weights=False deliberately."
        )

    lengths = {len(values) for values, _ in components.values()}
    if len(lengths) != 1:
        raise ValueError(f"components have differing lengths: {lengths}")

    score = np.zeros(lengths.pop(), dtype=float)
    for values, weight in components.values():
        score += np.asarray(values, dtype=float) * weight
    return score


def decile_table(df: pd.DataFrame, score_col: str = "score", label_col: str = "case",
                 n_bins: int = 10) -> pd.DataFrame:
    """Case rate by score decile, highest-scoring decile first.

    The single most informative table a score produces. A score that works
    shows a steep, monotone gradient from decile 1 down; a flat table means the
    score is not separating risk however good its headline metrics look.

    ``cumulative_recall`` against ``cumulative_alert_rate`` is the operating
    curve: alert on the top N deciles and you catch this share of cases for
    this share of the population.
    """
    if score_col not in df.columns:
        raise KeyError(f"column {score_col!r} not found; available: {list(df.columns)}")

    work = df[[score_col, label_col]].copy()
    # rank-then-cut rather than qcut on raw values: scores pile up on ties, and
    # qcut refuses duplicate bin edges rather than degrading gracefully.
    work["_rank"] = work[score_col].rank(method="first", ascending=False)
    work["decile"] = np.ceil(work["_rank"] / (len(work) / n_bins)).astype(int).clip(1, n_bins)

    grouped = work.groupby("decile")
    table = pd.DataFrame({
        "records": grouped.size(),
        "cases": grouped[label_col].sum(),
        "min_score": grouped[score_col].min(),
        "max_score": grouped[score_col].max(),
    })
    table["case_rate"] = table["cases"] / table["records"]
    table["cumulative_records"] = table["records"].cumsum()
    table["cumulative_cases"] = table["cases"].cumsum()
    table["cumulative_alert_rate"] = table["cumulative_records"] / len(work)
    table["cumulative_recall"] = table["cumulative_cases"] / max(work[label_col].sum(), 1)
    table["cumulative_precision"] = table["cumulative_cases"] / table["cumulative_records"]
    # >1 means the decile holds more risk than its share of the population.
    table["lift"] = table["case_rate"] / work[label_col].mean()
    return table.reset_index()


def score_cutoff_sweep(df: pd.DataFrame, cutoffs, score_col: str = "score",
                       label_col: str = "case") -> pd.DataFrame:
    """Metrics at each candidate score cut-off.

    The score analogue of a threshold sweep: everything scoring at or above the
    cut-off alerts.
    """
    labels = df[label_col].to_numpy()
    scores = df[score_col].to_numpy()
    rows = []
    for cutoff in cutoffs:
        rows.append({"cutoff": float(cutoff),
                     **classification_metrics(labels, scores >= cutoff)})
    return pd.DataFrame(rows)


def calibrate_probability(df: pd.DataFrame, score_col: str = "score", label_col: str = "case",
                          n_bins: int = 10) -> pd.DataFrame:
    """Observed case probability within each score band.

    Turns an arbitrary score into a probability estimate a reviewer can act on.
    "Score 72" means nothing on its own; "records scoring 70-80 were cases 9% of
    the time" is a statement an investigator and a validator can both use.

    ``ci_low``/``ci_high`` are Wilson intervals. Sparse bands at the top of the
    score range routinely carry intervals several times wider than the
    difference between neighbouring bands -- so a monotone-looking calibration
    table can be entirely consistent with a flat one.
    """
    from .challenger import wilson_interval

    work = df[[score_col, label_col]].copy()
    edges = np.linspace(work[score_col].min(), work[score_col].max() + 1e-9, n_bins + 1)
    work["band"] = pd.cut(work[score_col], bins=edges, include_lowest=True)

    rows = []
    for band, part in work.groupby("band", observed=True):
        successes, trials = int(part[label_col].sum()), len(part)
        low, high = wilson_interval(successes, trials) if trials else (np.nan, np.nan)
        rows.append({
            "band": str(band),
            "band_low": band.left,
            "records": trials,
            "cases": successes,
            "observed_probability": safe_divide(successes, trials),
            "ci_low": low,
            "ci_high": high,
        })
    return pd.DataFrame(rows).sort_values("band_low", ascending=False).reset_index(drop=True)


def optimise_cutoff(sweep: pd.DataFrame, objective: str = "recall",
                    max_alerts: int | None = None, min_precision: float | None = None,
                    strict: bool = True) -> pd.Series | None:
    """Pick a score cut-off under operational constraints.

    Same shape as :func:`~tmtuning.thresholds.optimise_threshold`, and for the
    same reason: a cut-off is a capacity decision before it is a statistical
    one. Ties break towards the higher cut-off -- equal detection for fewer
    alerts is strictly better.
    """
    if objective not in sweep.columns:
        raise KeyError(f"objective {objective!r} is not a column; available: {list(sweep.columns)}")

    feasible = sweep.copy()
    if max_alerts is not None:
        feasible = feasible[feasible["alerts"] <= max_alerts]
    if min_precision is not None:
        feasible = feasible[feasible["precision"] >= min_precision]
    feasible = feasible[feasible[objective].notna()]

    if feasible.empty:
        if strict:
            raise ValueError(
                "No score cut-off satisfies the constraints. Widen the candidate "
                "range, relax a constraint, or accept that the score cannot meet the target."
            )
        return None
    return feasible.sort_values([objective, "cutoff"], ascending=[False, False]).iloc[0]
