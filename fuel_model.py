from __future__ import annotations

"""
Negative Binomial model for main-engine fuel consumption.

Fits the number of main-engine frames per episode to a Negative Binomial
distribution via method-of-moments, then estimates P(FuelDepleted).

All math is implemented from scratch — no scipy.stats.

Parameterisation
----------------
X ~ NegBin(r, p)   counts *failures* before r successes.

    P(X = k) = C(k + r − 1, k) · p^r · (1 − p)^k

    Mean     = r(1 − p) / p
    Variance = r(1 − p) / p²

Method-of-moments:
    p = mean / variance
    r = mean² / (variance − mean)
"""

import math
from typing import Tuple

import numpy as np


def fit_negative_binomial(counts: np.ndarray) -> Tuple[float, float]:
    """
    Estimate (r, p) from a sample of counts via method-of-moments.

    Falls back to a Poisson-like default when the data is under-dispersed
    (variance ≤ mean).
    """
    counts = np.asarray(counts, dtype=np.float64)
    m  = float(np.mean(counts))
    s2 = float(np.var(counts, ddof=1)) if len(counts) > 1 else m + 1.0

    if s2 <= m or m <= 0:
        return (max(m, 1.0), 0.5)

    p = m / s2
    r = m ** 2 / (s2 - m)

    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    r = max(r, 0.1)
    return (r, p)


def _log_binom(n: float, k: int) -> float:
    """Logarithm of the generalised binomial coefficient C(n, k) for real n."""
    if k < 0:
        return -np.inf
    if k == 0:
        return 0.0
    total = 0.0
    for i in range(k):
        total += math.log(n - i) - math.log(i + 1)
    return total


def neg_binom_pmf(k: int, r: float, p: float) -> float:
    """
    Negative Binomial PMF computed from scratch:

        P(X = k) = C(k + r − 1, k) · p^r · (1 − p)^k
    """
    if k < 0 or p <= 0.0 or p >= 1.0 or r <= 0.0:
        return 0.0
    log_c = _log_binom(k + r - 1, k)
    log_p = r * math.log(p) + k * math.log(1.0 - p)
    return math.exp(log_c + log_p)


def prob_fuel_depleted(r: float, p: float, budget: int) -> float:
    """
    P(MainEngineFrames > budget)  =  1 − CDF(budget).

    Computes CDF by summing the PMF from 0 to *budget*.
    """
    cdf = 0.0
    for k in range(budget + 1):
        cdf += neg_binom_pmf(k, r, p)
        if cdf >= 1.0:
            return 0.0
    return max(0.0, 1.0 - cdf)
