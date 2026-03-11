from __future__ import annotations

"""
Poisson process utilities — implemented from scratch for CS109 demonstration.

Provides an explicit PMF and a Knuth inverse-CDF sampler.  Used in offline
analysis to model event arrival rates (e.g. main-engine firings per second
of game time).
"""

import math
import random


def poisson_pmf(k: int, lam: float) -> float:
    """
    Poisson probability mass function:

        P(K = k | λ) = e^{-λ} · λ^k / k!
    """
    if k < 0 or lam < 0.0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_sample(lam: float) -> int:
    """
    Draw K ~ Poisson(λ) using Knuth's product algorithm.

    Efficient for small λ, which fits our per-second event model.
    """
    if lam <= 0.0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1
