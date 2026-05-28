"""Arbiter is safety-critical: exhaustive unit coverage of the decision rules."""

from nse.config import SETTINGS
from nse.orchestrator import arbiter
from nse.orchestrator.schemas import BranchPrediction, PruneReason, Routing


def _pred(**kw) -> BranchPrediction:
    base = dict(branch_id="b", p_c=1, p_t=0.8, u=0.0, r_critic=0.0,
                r_long=0.0, c_planner=0.9)
    base.update(kw)
    return BranchPrediction(**base)


def test_symbolic_fail_prunes_before_scoring():
    pred = arbiter.decide(_pred(p_c=0))
    assert pred.routing == Routing.PRUNE
    assert pred.prune_reason == PruneReason.SYMBOLIC_FAIL
    assert pred.score is None


def test_high_uncertainty_never_prunes():
    # u above u_max must route to incremental sandbox, not prune.
    pred = arbiter.decide(_pred(u=SETTINGS.hp.u_max + 0.1))
    assert pred.routing == Routing.INCREMENTAL_SANDBOX
    assert pred.prune_reason is None


def test_low_score_prunes():
    pred = arbiter.decide(_pred(p_t=0.05, c_planner=0.1, r_critic=0.0))
    assert pred.routing == Routing.PRUNE
    assert pred.prune_reason == PruneReason.LOW_SCORE


def test_good_branch_executes():
    pred = arbiter.decide(_pred())
    assert pred.routing == Routing.EXECUTE
    assert pred.score is not None and pred.score >= SETTINGS.hp.tau_prune


def test_score_formula_matches_blueprint():
    p = _pred(p_t=0.8, c_planner=0.9, r_critic=0.1, r_long=0.2, u=0.04)
    hp = SETTINGS.hp
    u_norm = 0.04 / hp.max_variance
    expected = (1 * 0.8 * 0.9) - hp.lambda1 * 0.1 - hp.lambda2 * 0.2 \
        - hp.lambda3 * u_norm * (1 - 0.8)
    assert abs(arbiter.score(p) - expected) < 1e-9


def test_aggregate_p_t():
    assert abs(arbiter.aggregate_p_t(1.0, 0.0) - SETTINGS.hp.alpha) < 1e-9


def test_select_best_picks_highest_executable():
    a = arbiter.decide(_pred(branch_id="a", p_t=0.7))
    b = arbiter.decide(_pred(branch_id="b", p_t=0.95))
    c = arbiter.decide(_pred(branch_id="c", p_c=0))
    assert arbiter.select_best([a, b, c]).branch_id == "b"
