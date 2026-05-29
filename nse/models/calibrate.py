"""Calibration primitives — ECE, Brier, temperature scaling, isotonic.

These operate on logged (predicted p_t, observed tests_passed) pairs from the
DB. Retrain/recalibrate triggers (ECE > threshold, every N runs) are owned by
the orchestrator/CI; this module just computes and fits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
