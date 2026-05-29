"""Calibration + audit control loop (blueprint Phase 6).

The orchestrator logs every ``(predicted p_t, observed tests_passed)`` pair. This
module closes the loop:

* :func:`run_calibration` measures ECE/Brier on that logged history and, when the
  model is miscalibrated (ECE over threshold), fits an isotonic recalibrator and
  persists it. The orchestrator applies it to *future* predictions. Calibration
  always fits the *raw* logged ``p_t`` (the orchestrator logs pre-recalibration
  values), so re-running is idempotent — no composition drift.
* :func:`run_audit` samples the pruned-branch reservoir so a fraction of pruned
  decisions can be revisited (false-negative hunting), marking them sampled.

Every calibration run is recorded in the ``calibrations`` table for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from nse.config import SETTINGS
from nse.models.calibrate import (
    Recalibrator,
    _SKLEARN,
    compute_brier,
    compute_ece,
    fit_isotonic_recalibrator,
)
from nse.models.latent_model import WEIGHTS_DIR

if TYPE_CHECKING:  # avoid importing the DB layer at runtime / in tests of pure math
    from nse.db.db_client import DBClient

RECALIBRATOR_PATH = WEIGHTS_DIR / "recalibrator.json"


@dataclass
class CalibrationResult:
    n: int
    ece: float
    brier: float
    action: str       # "recalibrated_isotonic" | "ok" | "skipped_*"
    recalibrated: bool


# ───────────────────────────── calibration ─────────────────────────────


def run_calibration(
    db: "DBClient",
    ece_threshold: float | None = None,
    min_samples: int = 20,
    persist_path: Path | str = RECALIBRATOR_PATH,
) -> CalibrationResult:
    """Score logged predictions and recalibrate when ECE exceeds threshold."""
    ece_threshold = (
        SETTINGS.hp.ece_threshold if ece_threshold is None else ece_threshold
    )
    probs: list[float] = []
    labels: list[int] = []
    for r in db.predictions_with_outcomes():
        if r["p_t"] is not None and r["tests_passed"] is not None:
            probs.append(float(r["p_t"]))
            labels.append(int(r["tests_passed"]))

    n = len(labels)
    if n == 0:
        return CalibrationResult(0, 0.0, 0.0, "skipped_no_data", False)

    ece = compute_ece(probs, labels)
    brier = compute_brier(probs, labels)

    recalibrated = False
    if n < min_samples or len(set(labels)) < 2:
        action = "skipped_insufficient_data"
    elif ece > ece_threshold and _SKLEARN:
        fit_isotonic_recalibrator(probs, labels).save(persist_path)
        action = "recalibrated_isotonic"
        recalibrated = True
    else:
        action = "ok"

    db.insert_calibration(ece, brier, action)
    return CalibrationResult(n, ece, brier, action, recalibrated)


def load_recalibrator(
    path: Path | str = RECALIBRATOR_PATH,
) -> Optional[Recalibrator]:
    """Load the persisted recalibrator, or ``None`` if absent/unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return Recalibrator.load(path)
    except (ValueError, KeyError, OSError):
        return None


# ──────────────────────────── audit reservoir ───────────────────────────


def run_audit(
    db: "DBClient", sampling_percent: float | None = None
) -> list[dict]:
    """Sample a fraction of un-audited pruned branches and mark them sampled.

    Returns the sampled rows so a caller (CI / human / re-execution job) can
    revisit those pruned decisions. Re-execution to confirm false negatives is a
    heavier follow-up; this provides the reservoir sampling it builds on.
    """
    pct = (
        SETTINGS.hp.audit_sampling_percent
        if sampling_percent is None
        else sampling_percent
    )
    total = db.count_pruned_unsampled()
    if total == 0 or pct <= 0:  # pct<=0 disables sampling entirely
        return []
    limit = max(1, round(pct * total))
    sampled = db.sample_pruned_for_audit(limit)
    for row in sampled:
        db.mark_audited(int(row["id"]))
    return sampled
