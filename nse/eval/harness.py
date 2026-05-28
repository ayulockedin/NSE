"""Scoreboard for the latent transition model.

``evaluate_ensemble`` scores every labeled example with the ensemble's
``p_t_latent`` and reports calibration + discrimination metrics. ECE and Brier
are reused from :mod:`nse.models.calibrate` so there is one canonical
implementation; AUC uses scikit-learn when available.
"""

from __future__ import annotations

from dataclasses import dataclass

from nse.data.dataset import LabeledExample
from nse.models.calibrate import compute_brier, compute_ece
from nse.models.latent_model import LatentEnsemble

try:
    from sklearn.metrics import roc_auc_score

    _SKLEARN = True
except ImportError:  # pragma: no cover
    _SKLEARN = False


@dataclass
class EvalReport:
    n: int
    base_rate: float          # fraction of examples that actually pass
    brier: float              # lower is better (0 = perfect)
    ece: float                # lower is better (0 = perfectly calibrated)
    accuracy: float           # at threshold 0.5
    auc: float | None         # None when undefined (single class or no sklearn)
    threshold: float = 0.5

    def render(self) -> str:
        auc = "n/a" if self.auc is None else f"{self.auc:.3f}"
        return (
            "-- Latent model eval -------------------------------\n"
            f"  examples      : {self.n}\n"
            f"  base rate     : {self.base_rate:.3f}  (fraction passing)\n"
            f"  Brier         : {self.brier:.4f}   (lower better)\n"
            f"  ECE           : {self.ece:.4f}   (lower better)\n"
            f"  accuracy@{self.threshold:g}  : {self.accuracy:.3f}\n"
            f"  ROC-AUC       : {auc}\n"
            "----------------------------------------------------"
        )


def predict_probs(
    examples: list[LabeledExample], ensemble: LatentEnsemble | None = None
) -> list[float]:
    ensemble = ensemble or LatentEnsemble()
    return [
        ensemble.predict_from_features(ex.features).p_t_latent for ex in examples
    ]


def evaluate_ensemble(
    examples: list[LabeledExample],
    ensemble: LatentEnsemble | None = None,
    threshold: float = 0.5,
) -> EvalReport:
    if not examples:
        return EvalReport(0, 0.0, 0.0, 0.0, 0.0, None, threshold)

    probs = predict_probs(examples, ensemble)
    labels = [ex.label for ex in examples]
    n = len(labels)
    base_rate = sum(labels) / n

    correct = sum(
        1 for p, y in zip(probs, labels) if int(p >= threshold) == y
    )
    accuracy = correct / n

    auc: float | None = None
    if _SKLEARN and 0 < sum(labels) < n:  # AUC needs both classes present
        auc = float(roc_auc_score(labels, probs))

    return EvalReport(
        n=n,
        base_rate=base_rate,
        brier=compute_brier(probs, labels),
        ece=compute_ece(probs, labels),
        accuracy=accuracy,
        auc=auc,
        threshold=threshold,
    )
