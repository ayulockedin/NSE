"""Arbiter — deterministic scoring & pruning (pure Python, no LLM, no torch).

This is the safety-critical heart of NSE. It is intentionally simple and
side-effect free so it can be unit-tested exhaustively. It implements the
canonical decision rules (blueprint section 2):

    p_t      = alpha * p_t_latent + (1 - alpha) * p_t_sim
    u_norm   = u / max_variance
    S(B)     = (p_c * p_t * c_planner)
               - l1 * r_critic - l2 * r_long - l3 * u_norm * (1 - p_t)

Decision flow (per branch):
    symbolic_fail (p_c == 0)  -> PRUNE
    u > u_max                 -> INCREMENTAL_SANDBOX (never prune on ignorance)
    S < tau_prune             -> PRUNE
    otherwise                 -> EXECUTE

Immutable invariant: a branch is *never* pruned solely because u > u_max.
"""

from __future__ import annotations

from typing import Callable, Optional

from nse.config import SETTINGS
from nse.orchestrator.schemas import (
    BranchPrediction,
    PruneReason,
    Routing,
)


def aggregate_p_t(p_t_latent: float, p_t_sim: float, alpha: float | None = None) -> float:
    alpha = SETTINGS.hp.alpha if alpha is None else alpha
    return alpha * p_t_latent + (1.0 - alpha) * p_t_sim


def normalize_uncertainty(u: float) -> float:
    u_norm = u / SETTINGS.hp.max_variance
    return min(1.0, max(0.0, u_norm))


def score(pred: BranchPrediction, p_t: float | None = None) -> float:
    """Compute S(B). Assumes ``pred.p_t`` already aggregated.

    ``p_t`` overrides the value used in the score (e.g. a recalibrated p_t)
    without mutating ``pred.p_t``, which stays the raw logged value.
    """
    hp = SETTINGS.hp
    p_t = pred.p_t if p_t is None else p_t
    u_norm = normalize_uncertainty(pred.u)
    base = pred.p_c * p_t * pred.c_planner
    penalty = (
        hp.lambda1 * pred.r_critic
        + hp.lambda2 * pred.r_long
        + hp.lambda3 * u_norm * (1.0 - p_t)
    )
    return base - penalty


def decide(
    pred: BranchPrediction,
    recalibrator: Optional[Callable[[float], float]] = None,
) -> BranchPrediction:
    """Populate ``score``, ``routing`` and ``prune_reason`` on a prediction.

    Returns the same object (mutated) for convenient chaining. A ``recalibrator``
    (from the calibration loop) maps the aggregated ``p_t`` to a calibrated value
    used for scoring/routing only — ``pred.p_t`` keeps the raw value so the next
    calibration round fits raw->outcome and stays idempotent.
    """
    hp = SETTINGS.hp

    if pred.p_c == 0:
        pred.score = None
        pred.routing = Routing.PRUNE
        pred.prune_reason = PruneReason.SYMBOLIC_FAIL
        return pred

    p_t_eff = recalibrator(pred.p_t) if recalibrator is not None else pred.p_t
    pred.score = score(pred, p_t_eff)

    if pred.u > hp.u_max:
        # High epistemic uncertainty -> gather evidence, do NOT prune.
        pred.routing = Routing.INCREMENTAL_SANDBOX
        pred.prune_reason = None
        return pred

    if pred.score < hp.tau_prune:
        pred.routing = Routing.PRUNE
        pred.prune_reason = PruneReason.LOW_SCORE
        return pred

    pred.routing = Routing.EXECUTE
    pred.prune_reason = None
    return pred


def select_best(preds: list[BranchPrediction]) -> BranchPrediction | None:
    """Pick the highest-scoring branch routed to EXECUTE."""
    executable = [p for p in preds if p.routing == Routing.EXECUTE and p.score is not None]
    if not executable:
        return None
    return max(executable, key=lambda p: p.score)  # type: ignore[arg-type]
