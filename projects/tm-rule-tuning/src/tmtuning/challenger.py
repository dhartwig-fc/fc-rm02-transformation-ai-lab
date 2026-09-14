"""Champion vs challenger comparison and above/below-the-line testing.

Two distinct jobs live here.

*Champion/challenger* answers "is the proposed rule better than the one in
production?" -- and, critically, *what does it stop catching?* A challenger with
higher precision that misses a typology the incumbent covers is a downgrade
dressed up as an improvement.

*Above/below-the-line (ATL/BTL) testing* is the evidence a model validator or
regulator expects with a threshold change. Above the line, you sample alerts to
verify they are productive. Below the line, you sample the population the rule
does *not* alert on, to estimate how much risk you are choosing to miss -- with
a confidence interval, not an anecdote.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from .metrics import classification_metrics, safe_divide

__all__ = [
    "compare_rules",
    "rule_overlap",
    "wilson_interval",
    "required_sample_size",
    "atl_btl_sample",
    "btl_test",
]


def compare_rules(
    df: pd.DataFrame,
    rules: Mapping[str, np.ndarray | pd.Series],
    label_col: str = "case",
    champion: str | None = None,
) -> pd.DataFrame:
    """Score several rules on the same population.

    Parameters
    ----------
    rules:
        ``{name: boolean alert flag}``. Every flag must align with ``df``.
    champion:
        Name of the incumbent. When given, delta columns are added showing each
        challenger's movement against it.

    A comparison is only meaningful on an identical population and an identical
    label set. Running the champion on last year's data and the challenger on
    this year's is the single most common way a challenger is made to look good.
    """
    labels = df[label_col]
    rows = []
    for name, flags in rules.items():
        flags = np.asarray(flags)
        if flags.shape[0] != len(df):
            raise ValueError(f"rule {name!r} has {flags.shape[0]} flags for {len(df)} rows")
        rows.append({"rule": name, **classification_metrics(labels, flags)})

    out = pd.DataFrame(rows).set_index("rule")

    if champion is not None:
        if champion not in out.index:
            raise KeyError(f"champion {champion!r} is not among the rules: {list(out.index)}")
        base = out.loc[champion]
        for col in ("alerts", "tp", "fn", "precision", "recall"):
            out[f"delta_{col}"] = out[col] - base[col]
    return out


def rule_overlap(
    df: pd.DataFrame,
    champion: np.ndarray | pd.Series,
    challenger: np.ndarray | pd.Series,
    label_col: str = "case",
) -> pd.DataFrame:
    """Where champion and challenger agree and disagree, and what it costs.

    The headline metrics in :func:`compare_rules` net off gains against losses.
    This breaks the netting open into four cells:

    * **Both alert** -- the shared core.
    * **Champion only** -- risk the challenger would newly *miss*. Read the
      ``cases`` figure here before approving anything.
    * **Challenger only** -- risk the challenger newly catches.
    * **Neither** -- the blind spot both rules share.

    A challenger that nets +5 true cases while losing 20 in "champion only" and
    gaining 25 in "challenger only" is not a tweak -- it is a different rule
    detecting different behaviour, and it needs typology review, not just a
    metrics table.
    """
    champ = np.asarray(champion).astype(bool)
    chall = np.asarray(challenger).astype(bool)
    labels = np.asarray(df[label_col]).astype(bool)

    cells = {
        "Both alert": champ & chall,
        "Champion only": champ & ~chall,
        "Challenger only": ~champ & chall,
        "Neither alerts": ~champ & ~chall,
    }

    rows = []
    for name, mask in cells.items():
        n = int(mask.sum())
        cases = int((mask & labels).sum())
        rows.append({
            "cell": name, "rows": n, "cases": cases,
            "share_of_population": safe_divide(n, len(df)),
            "share_of_cases": safe_divide(cases, labels.sum()),
            "precision": safe_divide(cases, n),
        })
    return pd.DataFrame(rows)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Used instead of the textbook normal ("Wald") interval because BTL testing
    lives exactly where Wald breaks: small proportions, and samples where the
    observed count of productive items can be zero. Wald returns a zero-width
    interval ``[0, 0]`` when no successes are observed -- which would let a
    tuning paper claim with certainty that no risk sits below the line. Wilson
    returns a proper upper bound.

    >>> lo, hi = wilson_interval(0, 100)
    >>> round(lo, 4), round(hi, 4)
    (0.0, 0.037)
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    # Inverse normal CDF via the error function -- avoids a scipy dependency here.
    z = math.sqrt(2) * _erfinv(confidence)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    halfwidth = z / denom * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return max(centre - halfwidth, 0.0), min(centre + halfwidth, 1.0)


def _erfinv(x: float) -> float:
    """Inverse error function (Giles' rational approximation), accurate to ~1e-9."""
    w = -math.log((1.0 - x) * (1.0 + x))
    if w < 5.0:
        w -= 2.5
        coeffs = [2.81022636e-08, 3.43273939e-07, -3.5233877e-06, -4.39150654e-06,
                  0.00021858087, -0.00125372503, -0.00417768164, 0.246640727, 1.50140941]
    else:
        w = math.sqrt(w) - 3.0
        coeffs = [-0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844,
                  0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682]
    result = coeffs[0]
    for c in coeffs[1:]:
        result = result * w + c
    return result * x


def required_sample_size(expected_rate: float, margin_of_error: float, confidence: float = 0.95) -> int:
    """Sample size needed to estimate a proportion to a given precision.

    ``n = z^2 * p(1-p) / e^2``

    Plan the BTL sample with this *before* drawing it. A sample sized by
    convenience ("we reviewed 100") usually cannot distinguish a 1% below-the-
    line productive rate from a 5% one -- and that distinction is the entire
    question the test exists to answer.

    ``expected_rate`` of 0 or 1 is treated as 0.5, the variance-maximising
    (most conservative) assumption.
    """
    if not 0 < margin_of_error < 1:
        raise ValueError(f"margin_of_error must be in (0, 1), got {margin_of_error}")
    if not 0 <= expected_rate <= 1:
        raise ValueError(f"expected_rate must be in [0, 1], got {expected_rate}")

    p = 0.5 if expected_rate in (0.0, 1.0) else expected_rate
    z = math.sqrt(2) * _erfinv(confidence)
    return int(math.ceil(z**2 * p * (1 - p) / margin_of_error**2))


def atl_btl_sample(
    df: pd.DataFrame,
    alert_flag: np.ndarray | pd.Series,
    n_above: int = 100,
    n_below: int = 300,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Draw an above-the-line and a below-the-line review sample.

    Returns ``{"above": ..., "below": ...}``. Both are simple random samples
    without replacement, which is what makes the resulting rate estimate
    unbiased -- taking "the biggest 300 non-alerting customers" instead would
    estimate the miss rate among large customers, not below the line as a whole.

    If a real exercise does need to oversample a high-risk stratum, that is a
    stratified design and the estimate must be re-weighted accordingly; do not
    quote a stratified sample as if it were simple random.
    """
    flags = np.asarray(alert_flag).astype(bool)
    if flags.shape[0] != len(df):
        raise ValueError(f"alert_flag has {flags.shape[0]} entries for {len(df)} rows")

    above_pool = df[flags]
    below_pool = df[~flags]
    rng = np.random.default_rng(seed)

    def _draw(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(pool) == 0:
            return pool.copy()
        take = min(n, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        return pool.iloc[np.sort(idx)].copy()

    return {"above": _draw(above_pool, n_above), "below": _draw(below_pool, n_below)}


def btl_test(
    df: pd.DataFrame,
    alert_flag: np.ndarray | pd.Series,
    label_col: str = "case",
    n_below: int = 300,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Estimate missed risk below the line, with a confidence interval.

    Draws a BTL sample, counts how many sampled records were in fact true cases,
    and extrapolates to the whole below-the-line population.

    Returns the observed rate, its Wilson interval, and the implied number of
    missed cases across the full BTL population at the point estimate and at the
    upper bound.

    **Quote the upper bound.** A point estimate of "roughly 40 missed cases"
    invites a reviewer to ask what the number could plausibly be at worst; the
    upper confidence bound answers that question before it is asked, and is the
    figure risk appetite should actually be tested against.
    """
    flags = np.asarray(alert_flag).astype(bool)
    below_population = int((~flags).sum())
    if below_population == 0:
        raise ValueError("The rule alerts on the entire population; there is no below-the-line to test.")

    sample = atl_btl_sample(df, flags, n_above=0, n_below=n_below, seed=seed)["below"]
    reviewed = len(sample)
    productive = int(sample[label_col].sum())

    rate = safe_divide(productive, reviewed)
    lower, upper = wilson_interval(productive, reviewed, confidence)

    return {
        "below_the_line_population": below_population,
        "sampled": reviewed,
        "productive_in_sample": productive,
        "observed_rate": rate,
        "confidence": confidence,
        "rate_lower": lower,
        "rate_upper": upper,
        "estimated_missed_cases": rate * below_population,
        "estimated_missed_cases_upper": upper * below_population,
        "actual_missed_cases": int(df.loc[~flags, label_col].sum()),
    }
