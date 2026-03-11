from __future__ import annotations

"""
CS109-style validation metrics — all computed from scratch (no sklearn).

- Calibration bins  +  reliability diagram
- Brier score  and  Expected Calibration Error (ECE)
- Precision / Recall curve for safe-landing classification
- Expected-value vs. realised-return Pearson correlation
"""

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG


# ------------------------------------------------------------------ #
#  Data structures                                                    #
# ------------------------------------------------------------------ #
@dataclass
class CalibrationBin:
    lower: float
    upper: float
    mean_pred: float
    empirical_rate: float
    count: int


# ------------------------------------------------------------------ #
#  Calibration                                                        #
# ------------------------------------------------------------------ #
def compute_calibration_bins(
    pred_probs: np.ndarray,
    outcomes: np.ndarray,
) -> List[CalibrationBin]:
    """
    Bin predicted P(Crash) into K equal-width buckets and compute
    the empirical crash frequency in each bucket.

    outcomes: 1 = crash, 0 = safe
    """
    K = CONFIG.metrics.calib_bins
    edges = np.linspace(0.0, 1.0, K + 1)
    p = np.clip(pred_probs, 0.0, 1.0)
    y = outcomes.astype(np.float64)

    bins: List[CalibrationBin] = []
    for i in range(K):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (p >= lo) & (p < hi) if i < K - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append(CalibrationBin(lo, hi, 0.0, 0.0, 0))
        else:
            bins.append(CalibrationBin(
                lo, hi,
                mean_pred=float(p[mask].mean()),
                empirical_rate=float(y[mask].mean()),
                count=n,
            ))
    return bins


def brier_score(pred_probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score: mean( (p − y)² )."""
    return float(np.mean((pred_probs - outcomes) ** 2))


def expected_calibration_error(bins: List[CalibrationBin]) -> float:
    """Weighted average |mean_pred − empirical_rate| across bins."""
    total = sum(b.count for b in bins)
    if total == 0:
        return 0.0
    ece = sum(
        (b.count / total) * abs(b.mean_pred - b.empirical_rate)
        for b in bins if b.count > 0
    )
    return float(ece)


# ------------------------------------------------------------------ #
#  Precision / Recall                                                 #
# ------------------------------------------------------------------ #
def precision_recall_curve(
    pred_probs: np.ndarray,
    outcomes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep threshold τ over P(Crash) to classify Safe vs. Crash.

        Predicted Safe  if  P(Crash) < τ
        Predicted Crash if  P(Crash) ≥ τ

    Returns (thresholds, precisions, recalls) where
        Precision = P(actually Safe | Predicted Safe)
        Recall    = P(Predicted Safe | actually Safe)
    """
    taus = np.linspace(0.0, 1.0, CONFIG.metrics.pr_thresholds)
    safe_mask = outcomes == 0
    n_safe = int(safe_mask.sum())

    precisions: List[float] = []
    recalls:    List[float] = []

    for tau in taus:
        pred_safe = pred_probs < tau
        tp = int(np.logical_and(pred_safe,  safe_mask).sum())
        fp = int(np.logical_and(pred_safe, ~safe_mask).sum())
        fn = n_safe - tp

        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recalls.append(   tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    return taus, np.array(precisions), np.array(recalls)


# ------------------------------------------------------------------ #
#  EV vs. Return                                                      #
# ------------------------------------------------------------------ #
def compare_ev_vs_return(
    expected_values: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Pearson correlation between predicted EV and realised episode return."""
    if len(expected_values) < 2:
        return 0.0
    ev = np.asarray(expected_values, dtype=np.float64)
    rt = np.asarray(returns, dtype=np.float64)
    if np.std(ev) == 0.0 or np.std(rt) == 0.0:
        return 0.0
    return float(np.corrcoef(ev, rt)[0, 1])


# ------------------------------------------------------------------ #
#  Plotting helpers                                                   #
# ------------------------------------------------------------------ #
def plot_reliability_diagram(
    bins: List[CalibrationBin],
    path: str = "calibration.png",
) -> None:
    xs = [b.mean_pred for b in bins if b.count > 0]
    ys = [b.empirical_rate for b in bins if b.count > 0]
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(xs, ys, "o-", label="Empirical")
    ax.set_xlabel("Predicted P(Crash)")
    ax.set_ylabel("Empirical crash rate")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_precision_recall(
    precisions: np.ndarray,
    recalls: np.ndarray,
    path: str = "precision_recall.png",
) -> None:
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, "o-")
    ax.set_xlabel("Recall  P(Predicted Safe | Safe)")
    ax.set_ylabel("Precision  P(Safe | Predicted Safe)")
    ax.set_title("Precision–Recall Curve")
    ax.grid(True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
