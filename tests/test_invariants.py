"""Property tests for the safety invariants (Phase 12.3).

A randomized sweep asserts that ``arbiter.decide`` never violates an invariant
across the parameter space — the conformal-backed guarantee that the unit tests
can only spot-check.
"""

import random

from nse.config import SETTINGS
from nse.orchestrator import arbiter
from nse.orchestrator.invariants import (
    SafetyInvariantError,
    assert_safe,
    check_never_prune_on_uncertainty,
)
from nse.orchestrator.schemas import BranchPrediction, Routing


def _random_pred(rng: random.Random) -> BranchPrediction:
    return BranchPrediction(
        branch_id="b",
        p_c=rng.choice([0, 1]),
        p_t=rng.random(),
        u=rng.random() * 0.5,
        u_aleatoric=rng.random() * 0.25,
        r_critic=rng.random(),
        r_long=rng.random(),
        c_planner=rng.random(),
    )


def test_decide_never_violates_invariants_random_sweep():
    rng = random.Random(0)
    for _ in range(3000):
        pred = _random_pred(rng)
        cov = rng.choice([0.0, rng.random() * 0.3])
        conf = rng.choice([None, rng.random()])
        ale = rng.choice([None, SETTINGS.hp.aleatoric_max])
        arbiter.decide(pred, coverage_u=cov, conformal_threshold=conf, aleatoric_max=ale)
        # No recalibrator was passed, so the effective p_t equals pred.p_t.
        assert_safe(pred, coverage_u=cov, p_t_effective=pred.p_t, conformal_threshold=conf)


def test_high_uncertainty_low_score_never_low_score_prunes():
    pred = arbiter.decide(
        BranchPrediction(
            branch_id="b", p_c=1, p_t=0.02, u=SETTINGS.hp.u_max + 0.2, c_planner=0.1
        )
    )
    assert pred.routing != Routing.PRUNE  # routed to gather evidence instead
    assert check_never_prune_on_uncertainty(pred)


def test_assert_safe_catches_a_constructed_violation():
    # A hand-built inconsistent decision (symbolic fail but routed EXECUTE) must
    # be caught by the guard.
    bad = BranchPrediction(branch_id="b", p_c=0, routing=Routing.EXECUTE)
    try:
        assert_safe(bad)
        raise AssertionError("expected SafetyInvariantError")
    except SafetyInvariantError:
        pass


def test_conformal_invariant_holds_in_sweep():
    # EXECUTE never occurs below the conformal threshold.
    rng = random.Random(7)
    for _ in range(1000):
        pred = _random_pred(rng)
        pred.p_c = 1
        threshold = 0.6
        arbiter.decide(pred, conformal_threshold=threshold)
        if pred.routing == Routing.EXECUTE:
            assert pred.p_t >= threshold
