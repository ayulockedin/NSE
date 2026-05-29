"""Conformal-backed safety invariants (Phase 12.3).

The arbiter's decision rules encode safety properties that must hold for *every*
input, not only the cases a unit test happened to pick. This module states them
explicitly as checkable predicates and provides a defense-in-depth guard the
orchestrator asserts after each decision — turning "never prune on uncertainty"
from a comment into an *enforced*, property-tested invariant.

Invariants:
  I1  Never prune on uncertainty: if effective uncertainty exceeds ``u_max``, the
      branch is routed to gather evidence / escalated, never PRUNEd **for low
      score**. (Pruning under high uncertainty is allowed only for evidence- or
      safety-based reasons: symbolic failure, unsafe patch, oracle crash.)
  I2  Symbolic failure always prunes: ``p_c == 0`` -> PRUNE (never execute code
      that failed the compile/type gate).
  I3  Conformal gate: a branch is never routed EXECUTE with a calibrated ``p_t``
      below the conformal threshold (when one is set) — the distribution-free
      false-execute guarantee from Phase 7.2.
"""

from __future__ import annotations

from typing import Optional

from nse.config import SETTINGS
from nse.orchestrator.schemas import BranchPrediction, PruneReason, Routing


class SafetyInvariantError(AssertionError):
    """A decision violated a safety invariant — a logic bug in the decision path."""


def check_never_prune_on_uncertainty(pred: BranchPrediction, coverage_u: float = 0.0) -> bool:
    """I1: a high-uncertainty branch is never pruned *for low score*."""
    u_eff = max(pred.u, coverage_u)
    if (
        u_eff > SETTINGS.hp.u_max
        and pred.routing == Routing.PRUNE
        and pred.prune_reason == PruneReason.LOW_SCORE
    ):
        return False
    return True


def check_symbolic_fail_prunes(pred: BranchPrediction) -> bool:
    """I2: a symbolic-gate failure (p_c == 0) is always pruned."""
    return pred.routing == Routing.PRUNE if pred.p_c == 0 else True


def check_conformal_execute(
    pred: BranchPrediction, p_t_effective: float, conformal_threshold: Optional[float]
) -> bool:
    """I3: never EXECUTE below the conformal threshold (when one is set)."""
    if conformal_threshold is not None and pred.routing == Routing.EXECUTE:
        return p_t_effective >= conformal_threshold
    return True


def assert_safe(
    pred: BranchPrediction,
    coverage_u: float = 0.0,
    p_t_effective: Optional[float] = None,
    conformal_threshold: Optional[float] = None,
) -> BranchPrediction:
    """Raise :class:`SafetyInvariantError` if ``pred`` violates a safety invariant.
    Returns ``pred`` for chaining. Cheap enough to run after every decision."""
    if not check_never_prune_on_uncertainty(pred, coverage_u):
        raise SafetyInvariantError(
            f"I1 violated: LOW_SCORE prune under uncertainty "
            f"(u={pred.u}, cov={coverage_u} > u_max={SETTINGS.hp.u_max})"
        )
    if not check_symbolic_fail_prunes(pred):
        raise SafetyInvariantError(f"I2 violated: p_c==0 not pruned (routing={pred.routing})")
    pte = pred.p_t if p_t_effective is None else p_t_effective
    if not check_conformal_execute(pred, pte, conformal_threshold):
        raise SafetyInvariantError(
            f"I3 violated: EXECUTE with p_t {pte} below conformal {conformal_threshold}"
        )
    return pred
