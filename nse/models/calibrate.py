"""Calibration primitives — ECE, Brier, temperature scaling, isotonic.

These operate on logged (predicted p_t, observed tests_passed) pairs from the
DB. Retrain/recalibrate triggers (ECE > threshold, every N runs) are owned by
the orchestrator/CI; this module just computes and fits.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression

    _SKLEARN = True
except ImportError:  # pragma: no cover
    _SKLEARN = False


def compute_ece(prob: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    prob_a = np.asarray(prob, dtype=float)
    labels_a = np.asarray(labels, dtype=float)
    if prob_a.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (prob_a >= bins[i]) & (prob_a < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels_a[mask].mean()
        conf = prob_a[mask].mean()
        ece += (mask.sum() / prob_a.size) * abs(acc - conf)
    return float(ece)


def compute_brier(prob: Sequence[float], labels: Sequence[int]) -> float:
    """Brier score = mean squared error of probabilistic predictions."""
    prob_a = np.asarray(prob, dtype=float)
    labels_a = np.asarray(labels, dtype=float)
    if prob_a.size == 0:
        return 0.0
    return float(np.mean((prob_a - labels_a) ** 2))


def fit_isotonic(prob: Sequence[float], labels: Sequence[int]):
    """Fit a non-parametric recalibrator. Returns a callable p -> p'."""
    if not _SKLEARN:
        raise RuntimeError("scikit-learn not installed")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(prob, dtype=float), np.asarray(labels, dtype=float))
    return lambda p: float(iso.predict([p])[0])


def fit_temperature(logits: Sequence[float], labels: Sequence[int], lr: float = 0.01,
                    steps: int = 500) -> float:
    """Fit a scalar temperature T minimising NLL. Pure-numpy gradient descent.

    Returns T; apply as sigmoid(logit / T).
    """
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=float)
    if z.size == 0:
        return 1.0
    log_t = 0.0
    for _ in range(steps):
        t = np.exp(log_t)
        p = 1.0 / (1.0 + np.exp(-z / t))
        # d(NLL)/d(log_t)
        grad = np.mean((p - y) * (z / t))
        log_t -= lr * grad
    return float(np.exp(log_t))
