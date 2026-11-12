"""Synthetic transaction-monitoring populations for tuning exercises.

Two generators, for two different jobs:

``spec_population``
    The Week 2 generator exactly as written in the course spec. Its label is
    drawn *independently* of the amount. That is not a bug to fix -- it is the
    week's most useful lesson, and the docstring explains why.

``generate_population``
    A risk-linked population where the probability of being a true case really
    does depend on transaction behaviour. Everything from Week 3 onwards needs
    this, because a threshold can only be tuned if the risk signal is there to
    be found.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "SEGMENTS",
    "CUSTOMER_RISK_LEVELS",
    "JURISDICTION_TIERS",
    "SegmentProfile",
    "spec_population",
    "generate_population",
    "split_by_period",
]


@dataclass(frozen=True)
class SegmentProfile:
    """Behavioural profile for one customer segment.

    ``velocity_lambda`` is the Poisson mean for the count of *outbound wire
    transactions* in the period -- deliberately much smaller than ``txn_lambda``
    (which counts all transactions), so that a challenger condition such as
    "velocity > 5" is genuinely selective rather than true for almost everyone.

    ``wire_scale`` and ``cash_scale`` are gamma scale parameters in GBP; the
    shape is fixed at 2.0 so every segment keeps the same right-skewed shape and
    only its magnitude differs. ``risk_offset`` is an additive term on the
    log-odds of being a true case -- it is what makes segment-level calibration
    (Week 5) produce different answers per segment rather than one global one.
    """

    weight: float
    wire_scale: float
    cash_scale: float
    txn_lambda: float
    velocity_lambda: float
    high_risk_jurisdiction_p: float
    pep_p: float
    risk_offset: float


#: Four segments spanning three orders of magnitude of transaction value.
#: A single global threshold cannot serve all four -- which is the point.
SEGMENTS: Mapping[str, SegmentProfile] = {
    "RETAIL": SegmentProfile(
        weight=0.70, wire_scale=3_000, cash_scale=1_500, txn_lambda=18, velocity_lambda=1.6,
        high_risk_jurisdiction_p=0.03, pep_p=0.002, risk_offset=-0.35,
    ),
    "SME": SegmentProfile(
        weight=0.20, wire_scale=15_000, cash_scale=8_000, txn_lambda=45, velocity_lambda=3.0,
        high_risk_jurisdiction_p=0.08, pep_p=0.010, risk_offset=0.20,
    ),
    "CORPORATE": SegmentProfile(
        weight=0.08, wire_scale=60_000, cash_scale=5_000, txn_lambda=140, velocity_lambda=4.5,
        high_risk_jurisdiction_p=0.12, pep_p=0.020, risk_offset=0.10,
    ),
    "PRIVATE": SegmentProfile(
        weight=0.02, wire_scale=40_000, cash_scale=20_000, txn_lambda=25, velocity_lambda=2.2,
        high_risk_jurisdiction_p=0.20, pep_p=0.120, risk_offset=0.80,
    ),
}

# Log-odds weights on the standardised risk drivers. Deliberately modest: a
# real rule separates risk imperfectly, and a tuning exercise on perfectly
# separable data teaches nothing.
_BETA = {
    "wire": 0.90,
    "cash_ratio": 0.70,
    "velocity": 0.55,
    "customer_risk": 0.65,
    "jurisdiction": 1.10,
    "pep": 0.90,
    "new_account": 0.60,
}

#: Ordinal encoding of the KYC customer risk rating, used as a model driver.
CUSTOMER_RISK_LEVELS = {"LOW": 0.0, "MEDIUM": 1.0, "HIGH": 2.0}

#: Jurisdiction risk tiers and their relative weight on the log-odds of risk.
#: Tiered rather than binary because a geo score (Week 8) needs a gradient --
#: and because "high risk country" is itself a tiered judgement in any real
#: country risk methodology, not an on/off flag.
JURISDICTION_TIERS = {"DOMESTIC": 0.0, "STANDARD": 0.3, "ELEVATED": 0.7, "HIGH": 1.0}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(linear_term: np.ndarray, target_prevalence: float) -> float:
    """Find the intercept that makes mean P(case) equal the target prevalence.

    Bisection on a monotone function -- the mean of a sigmoid is strictly
    increasing in the intercept, so 60 halvings is far more than enough.
    """
    lo, hi = -40.0, 40.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(linear_term + mid).mean() < target_prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def spec_population(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """The Week 2 population, reproduced exactly as the course spec defines it.

    .. code-block:: python

        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "amount": np.random.gamma(2, 20000, n),
            "case":   np.random.binomial(1, 0.04, n),
        })

    **Read this before you interpret your Week 2 results.** ``case`` is drawn
    from a fixed 4% binomial with no reference to ``amount``. Risk and value are
    statistically independent. The consequence is that precision is flat at
    roughly 4% at *every* threshold you test -- £25k, £50k, £75k, £100k -- while
    recall falls as the threshold rises. Raising the threshold therefore buys no
    precision and costs pure detection.

    That flat line is the correct answer, and recognising it is the skill. A
    threshold sweep that shows no precision lift is telling you the variable you
    are tuning carries no risk signal in this sample. In a real tuning exercise
    that finding kills the rule (or the label set) -- it does not mean you
    should keep sliding the threshold until a number looks better.

    Week 3 onwards uses :func:`generate_population`, where the signal exists.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "amount": rng.gamma(2, 20000, n),
            "case": rng.binomial(1, 0.04, n),
        }
    )


def generate_population(
    n: int = 20_000,
    seed: int = 42,
    prevalence: float = 0.03,
    n_periods: int = 12,
    start_period: str = "2025-01",
    drift_strength: float = 0.0,
) -> pd.DataFrame:
    """Generate a risk-linked monthly monitoring population.

    Each row is one customer-month: the unit a periodic TM rule actually scores.

    Parameters
    ----------
    n:
        Number of customer-month rows.
    seed:
        Seed for the random generator, so every exercise is reproducible.
    prevalence:
        Target share of rows that are true cases. AML realistic range is roughly
        1-5%; the default 3% keeps the class imbalance honest.
    n_periods:
        Number of consecutive monthly periods to spread the rows across. Needed
        for the Week 7 stability and drift work.
    start_period:
        First month, as ``YYYY-MM``.
    drift_strength:
        0.0 gives a stationary population. Values above 0 progressively inflate
        transaction values and tilt the segment mix towards higher-value
        segments as the periods advance -- the classic cause of alert-volume
        creep and precision decay that Week 7 asks you to detect.

    Returns
    -------
    DataFrame with columns:
        ``customer_id``, ``period``, ``period_index``, ``segment``,
        ``monthly_wire_value``, ``monthly_cash_deposits``, ``txn_count``,
        ``velocity``, ``cash_ratio``, ``customer_risk``, ``jurisdiction_risk``,
        ``high_risk_jurisdiction``, ``pep_flag``, ``account_age_months``,
        ``risk_probability``, ``case``.

    ``velocity`` is the count of outbound wire transactions in the period,
    ``customer_risk`` the KYC rating (LOW/MEDIUM/HIGH), and ``jurisdiction_risk``
    the country tier (DOMESTIC/STANDARD/ELEVATED/HIGH). All three are real risk
    drivers in the generator, so a challenger rule (Week 6) or a weighted score
    (Week 8) built on them finds genuine signal rather than noise.

    ``high_risk_jurisdiction`` is kept as a binary derived from the top tier, so
    rules written against the flag behave exactly as before.

    Notes
    -----
    ``risk_probability`` is the generative truth. It is included so you can
    reason about what an ideal rule *could* achieve, but a rule you tune must
    never consume it -- that would be target leakage, and every metric it
    produced would be fiction.
    """
    if not 0 < prevalence < 1:
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")
    if n_periods < 1:
        raise ValueError(f"n_periods must be at least 1, got {n_periods}")

    rng = np.random.default_rng(seed)

    period_index = rng.integers(0, n_periods, n)
    # Drift ramps linearly from 0 in the first period to `drift_strength` in the last.
    drift = drift_strength * (period_index / max(n_periods - 1, 1))

    names = list(SEGMENTS)
    base_weights = np.array([SEGMENTS[s].weight for s in names])

    # Under drift, the mix tilts towards the higher-value segments (later names).
    tilt = np.linspace(0.0, 1.0, len(names))
    segment_idx = np.empty(n, dtype=int)
    for p in range(n_periods):
        mask = period_index == p
        if not mask.any():
            continue
        step = drift_strength * (p / max(n_periods - 1, 1))
        weights = base_weights * (1.0 + step * tilt)
        weights = weights / weights.sum()
        segment_idx[mask] = rng.choice(len(names), size=int(mask.sum()), p=weights)

    segment = np.array(names)[segment_idx]
    wire_scale = np.array([SEGMENTS[s].wire_scale for s in names])[segment_idx]
    cash_scale = np.array([SEGMENTS[s].cash_scale for s in names])[segment_idx]
    txn_lambda = np.array([SEGMENTS[s].txn_lambda for s in names])[segment_idx]
    velocity_lambda = np.array([SEGMENTS[s].velocity_lambda for s in names])[segment_idx]
    hrj_p = np.array([SEGMENTS[s].high_risk_jurisdiction_p for s in names])[segment_idx]
    pep_p = np.array([SEGMENTS[s].pep_p for s in names])[segment_idx]
    risk_offset = np.array([SEGMENTS[s].risk_offset for s in names])[segment_idx]

    # Value inflation from drift on top of the segment mix shift.
    inflation = 1.0 + 0.45 * drift
    monthly_wire_value = rng.gamma(2.0, wire_scale * inflation)
    monthly_cash_deposits = rng.gamma(2.0, cash_scale * inflation)
    txn_count = rng.poisson(txn_lambda) + 1
    velocity = rng.poisson(velocity_lambda)

    # Jurisdiction tier. The HIGH tier keeps each segment's original high-risk
    # probability, so `high_risk_jurisdiction` below is unchanged; the ELEVATED
    # tier sits underneath it and carries part of the risk a binary flag misses.
    tier_names = list(JURISDICTION_TIERS)
    jurisdiction_risk = np.empty(n, dtype=object)
    draw = rng.random(n)
    for i in range(n):
        p_high = hrj_p[i]
        p_elevated = min(p_high * 2.5, 0.35)
        if draw[i] < p_high:
            jurisdiction_risk[i] = "HIGH"
        elif draw[i] < p_high + p_elevated:
            jurisdiction_risk[i] = "ELEVATED"
        elif draw[i] < p_high + p_elevated + 0.35:
            jurisdiction_risk[i] = "STANDARD"
        else:
            jurisdiction_risk[i] = "DOMESTIC"

    jurisdiction_weight = np.vectorize(JURISDICTION_TIERS.__getitem__)(jurisdiction_risk)
    high_risk_jurisdiction = (jurisdiction_risk == "HIGH").astype(int)
    pep_flag = rng.binomial(1, pep_p)
    account_age_months = rng.integers(1, 180, n)

    # KYC customer risk rating, as assigned at onboarding and periodic review.
    # Driven by the same jurisdiction and PEP factors an analyst would use, plus
    # noise -- a real rating is a judgement, so it is correlated with risk
    # without being a clean function of it.
    rating_latent = (
        1.3 * high_risk_jurisdiction
        + 1.5 * pep_flag
        + 0.5 * (account_age_months < 12)
        + rng.normal(0, 1.0, n)
    )
    cut_medium, cut_high = np.quantile(rating_latent, [0.72, 0.93])
    customer_risk = np.where(rating_latent >= cut_high, "HIGH",
                             np.where(rating_latent >= cut_medium, "MEDIUM", "LOW"))
    customer_risk_ordinal = np.vectorize(CUSTOMER_RISK_LEVELS.__getitem__)(customer_risk)

    total_value = monthly_wire_value + monthly_cash_deposits
    cash_ratio = monthly_cash_deposits / np.maximum(total_value, 1.0)

    def _z(x: np.ndarray) -> np.ndarray:
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)

    linear = (
        _BETA["wire"] * _z(np.log1p(monthly_wire_value))
        + _BETA["cash_ratio"] * _z(cash_ratio)
        + _BETA["velocity"] * _z(velocity.astype(float))
        + _BETA["customer_risk"] * customer_risk_ordinal
        + _BETA["jurisdiction"] * jurisdiction_weight
        + _BETA["pep"] * pep_flag
        + _BETA["new_account"] * (account_age_months < 12).astype(float)
        + risk_offset
    )
    intercept = _solve_intercept(linear, prevalence)
    risk_probability = _sigmoid(linear + intercept)
    case = rng.binomial(1, risk_probability)

    periods = pd.period_range(start=start_period, periods=n_periods, freq="M")

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "period": periods[period_index].astype(str),
            "period_index": period_index,
            "segment": segment,
            "monthly_wire_value": monthly_wire_value.round(2),
            "monthly_cash_deposits": monthly_cash_deposits.round(2),
            "txn_count": txn_count,
            "velocity": velocity,
            "cash_ratio": cash_ratio.round(4),
            "customer_risk": customer_risk,
            "jurisdiction_risk": jurisdiction_risk,
            "high_risk_jurisdiction": high_risk_jurisdiction,
            "pep_flag": pep_flag,
            "account_age_months": account_age_months,
            "risk_probability": risk_probability.round(5),
            "case": case,
        }
    )
    return frame.sort_values(["period_index", "customer_id"]).reset_index(drop=True)


def split_by_period(
    df: pd.DataFrame, holdout_periods: int = 3, period_col: str = "period_index"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a population into an in-time tuning sample and an out-of-time holdout.

    Splitting on *time* rather than at random is the whole point. A random split
    lets the tuning sample and the validation sample share the same months, so a
    threshold that only works in one quarter's conditions still validates
    cleanly. An out-of-time holdout is what a model validator will ask for.
    """
    if holdout_periods < 1:
        raise ValueError(f"holdout_periods must be at least 1, got {holdout_periods}")
    cutoff = df[period_col].max() - holdout_periods + 1
    if cutoff <= df[period_col].min():
        raise ValueError(
            f"holdout_periods={holdout_periods} leaves no in-time data; "
            f"population spans {df[period_col].min()}-{df[period_col].max()}"
        )
    return df[df[period_col] < cutoff].copy(), df[df[period_col] >= cutoff].copy()
