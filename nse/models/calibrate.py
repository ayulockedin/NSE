"""Calibration primitives — ECE, Brier, temperature scaling, isotonic.

These operate on logged (predicted p_t, observed tests_passed) pairs from the
DB. Retrain/recalibrate triggers (ECE > threshold, every N runs) are owned by
the orchestrator/CI; this module just computes and fits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression

    _SKLEARN = True
except ImportError:  # pragma: no cover
    _SKLEARN = False

try:
    from scipy.stats import beta as _beta

    _SCIPY = True
except ImportError:  # pragma: no cover
    _SCIPY = False


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


@dataclass
class Recalibrator:
    """A serializable monotone map ``p -> p'`` stored as a sampled grid.

    Sampling the fitted isotonic curve onto a fixed ``[0,1]`` grid and applying
    it with ``np.interp`` avoids pickling a scikit-learn estimator (fragile
    across versions) — the persisted artifact is just two float lists.
    """

    xs: list[float]
    ys: list[float]

    def __call__(self, p: float) -> float:
        return float(np.interp(p, self.xs, self.ys))

    def to_json(self) -> str:
        return json.dumps({"xs": self.xs, "ys": self.ys})

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, text: str) -> "Recalibrator":
        d = json.loads(text)
        return cls(xs=list(d["xs"]), ys=list(d["ys"]))

    @classmethod
    def load(cls, path: Path | str) -> "Recalibrator":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def fit_isotonic_recalibrator(
    prob: Sequence[float], labels: Sequence[int], grid: int = 101
) -> Recalibrator:
    """Fit isotonic regression and sample it onto a grid -> a :class:`Recalibrator`."""
    if not _SKLEARN:
        raise RuntimeError("scikit-learn not installed")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(prob, dtype=float), np.asarray(labels, dtype=float))
    xs = np.linspace(0.0, 1.0, grid)
    ys = iso.predict(xs)
    return Recalibrator(xs.tolist(), [float(y) for y in ys])


# ───────────────────── conformal EXECUTE gate (Phase 7.2) ─────────────────


# A threshold above 1.0 is the "execute nothing" sentinel: no p_t can clear it,
# so when the false-execute guarantee is unachievable the gate stays conservative.
CONFORMAL_NEVER = 1.01


def binomial_upper_bound(k: int, n: int, delta: float) -> float:
    """Upper (1-delta) confidence bound on a failure probability given k failures
    in n trials. Clopper-Pearson (exact, via scipy) when available, else the
    distribution-free Hoeffding bound. Both are valid finite-sample bounds."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    if _SCIPY:
        return float(_beta.ppf(1.0 - delta, k + 1, n - k))
    return min(1.0, k / n + math.sqrt(math.log(1.0 / delta) / (2 * n)))


def fit_conformal_threshold(
    prob: Sequence[float],
    labels: Sequence[int],
    alpha: float,
    delta: float,
    min_samples: int = 20,
    min_support: int = 5,
) -> Optional[float]:
    """Smallest p_t threshold τ such that EXECUTE-ing at ``p_t >= τ`` keeps the
    false-execute rate ≤ ``alpha`` with confidence ``1 - delta``.

    Distribution-free: validity rests on a binomial tail bound over the held-out
    calibration set, no distributional assumptions. Returns the smallest valid τ
    (max coverage); ``CONFORMAL_NEVER`` if no τ can be guaranteed; ``None`` when
    there isn't enough calibration data to make a claim (gate stays disabled).
    """
    probs = list(prob)
    labs = list(labels)
    n_total = len(labs)
    if n_total < min_samples or len(set(labs)) < 2:
        return None

    for tau in sorted(set(probs)):  # ascending -> first valid is the most permissive
        executed = [(p, y) for p, y in zip(probs, labs) if p >= tau]
        n = len(executed)
        if n < min_support:
            continue
        failures = sum(1 for _, y in executed if y == 0)
        if binomial_upper_bound(failures, n, delta) <= alpha:
            return float(tau)
    return CONFORMAL_NEVER


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
