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


def score(
    pred: BranchPrediction, p_t: float | None = None, u: float | None = None
) -> float:
    """Compute S(B). Assumes ``pred.p_t`` already aggregated.

    ``p_t`` / ``u`` override the values used in the score (e.g. a recalibrated p_t
    or a coverage-inflated uncertainty) without mutating ``pred``, which keeps the
    raw logged values.
    """
    hp = SETTINGS.hp
    p_t = pred.p_t if p_t is None else p_t
    u = pred.u if u is None else u
    u_norm = normalize_uncertainty(u)
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
    coverage_u: float = 0.0,
    conformal_threshold: Optional[float] = None,
    aleatoric_max: Optional[float] = None,
) -> BranchPrediction:
    """Populate ``score``, ``routing`` and ``prune_reason`` on a prediction.

    Returns the same object (mutated) for convenient chaining.

    ``recalibrator`` (calibration loop) maps the aggregated ``p_t`` to a
    calibrated value used for scoring/routing only — ``pred.p_t`` keeps the raw
    value so re-calibration stays idempotent.

    ``coverage_u`` (Phase 7.1) is the uncertainty that the change's edited lines
    are under-tested. It combines with the model's epistemic ``pred.u`` as
    ``max(pred.u, coverage_u)`` for routing + scoring, so an under-tested change
    is sent to gather evidence rather than trusted — extending the
    "never prune on uncertainty" invariant. ``pred.u`` stays the model's value.

    ``conformal_threshold`` (Phase 7.2) is the conformal EXECUTE gate: a branch
    that clears the score gate but whose calibrated ``p_t`` is below the threshold
    lacks the statistical guarantee to act on, so it routes to
    ``INCREMENTAL_SANDBOX`` (gather more evidence) instead of EXECUTE.

    ``aleatoric_max`` (Phase 12.1) escalates a would-EXECUTE branch to
    ``HUMAN_REVIEW`` when its *irreducible* uncertainty ``pred.u_aleatoric``
    exceeds it **and** the aggregate decision is itself a near-coin-flip
    (``|p_t - 0.5| <= coin_flip_band``): epistemic is already low here, so more
    evidence can't help and a human should decide. A confident aggregate is
    unaffected even if the latent model alone was uncertain.
    """
    hp = SETTINGS.hp

    if pred.p_c == 0:
        pred.score = None
        pred.routing = Routing.PRUNE
        pred.prune_reason = PruneReason.SYMBOLIC_FAIL
        return pred

    p_t_eff = recalibrator(pred.p_t) if recalibrator is not None else pred.p_t
    u_eff = max(pred.u, coverage_u)
    pred.score = score(pred, p_t_eff, u_eff)

    if u_eff > hp.u_max:
        # High epistemic OR coverage uncertainty -> gather evidence, do NOT prune.
        pred.routing = Routing.INCREMENTAL_SANDBOX
        pred.prune_reason = None
        return pred

    if pred.score < hp.tau_prune:
        pred.routing = Routing.PRUNE
        pred.prune_reason = PruneReason.LOW_SCORE
        return pred

    if conformal_threshold is not None and p_t_eff < conformal_threshold:
        # Scored OK, but below the conformal guarantee -> gather more evidence
        # rather than EXECUTE on an unguaranteed prediction.
        pred.routing = Routing.INCREMENTAL_SANDBOX
        pred.prune_reason = None
        return pred

    if (
        aleatoric_max is not None
        and pred.u_aleatoric > aleatoric_max
        and abs(p_t_eff - 0.5) <= hp.coin_flip_band
    ):
        # Irreducible (aleatoric) near-coin-flip: epistemic is already low here, so
        # more evidence won't help, and the *aggregate* call is itself ~50/50 -> a
        # human decides rather than auto-executing. A confident aggregate (e.g. a
        # confident LLM resolving an uncertain latent) is unaffected.
        pred.routing = Routing.HUMAN_REVIEW
        pred.prune_reason = None
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
