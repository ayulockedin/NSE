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


def test_low_coverage_routes_to_incremental_not_execute():
    # A confident, well-scored branch would EXECUTE...
    assert arbiter.decide(_pred()).routing == Routing.EXECUTE
    # ...but if its edited lines are under-tested (coverage_u > u_max), it must
    # route to gather evidence instead — never silently EXECUTE on weak evidence.
    pred = _pred()
    decided = arbiter.decide(pred, coverage_u=SETTINGS.hp.u_max + 0.05)
    assert decided.routing == Routing.INCREMENTAL_SANDBOX
    assert pred.u == 0.0  # the model's epistemic u is left untouched (logged raw)


def test_high_coverage_leaves_execute_unchanged():
    # Fully-covered change (coverage_u = 0) doesn't alter the decision.
    assert arbiter.decide(_pred(), coverage_u=0.0).routing == Routing.EXECUTE


def test_conformal_gate_blocks_unguaranteed_execute():
    # p_t below the conformal threshold: scored OK but no guarantee -> gather more
    # evidence instead of EXECUTE.
    pred = arbiter.decide(_pred(p_t=0.8), conformal_threshold=0.9)
    assert pred.routing == Routing.INCREMENTAL_SANDBOX
    # p_t at/above the threshold executes normally.
    assert arbiter.decide(_pred(p_t=0.8), conformal_threshold=0.5).routing == Routing.EXECUTE


def test_conformal_gate_applies_after_score_gate():
    # A low-score branch still PRUNEs; the conformal gate doesn't override that.
    pred = arbiter.decide(_pred(p_t=0.05, c_planner=0.1), conformal_threshold=0.0)
    assert pred.routing == Routing.PRUNE


def test_aleatoric_coinflip_escalates_to_human_review():
    # A would-EXECUTE near-coin-flip (p_t≈0.5) with high irreducible uncertainty:
    # with the gate set it escalates to HUMAN_REVIEW; without it, it EXECUTEs.
    assert arbiter.decide(_pred(p_t=0.5, u_aleatoric=0.24)).routing == Routing.EXECUTE
    pred = arbiter.decide(_pred(p_t=0.5, u_aleatoric=0.24), aleatoric_max=0.18)
    assert pred.routing == Routing.HUMAN_REVIEW
    assert pred.prune_reason is None


def test_confident_aggregate_executes_despite_high_aleatoric():
    # High latent aleatoric but a confident aggregate (p_t=0.8, not a coin-flip)
    # still executes -- the gate only catches genuine ~50/50 decisions.
    pred = arbiter.decide(_pred(p_t=0.8, u_aleatoric=0.24), aleatoric_max=0.18)
    assert pred.routing == Routing.EXECUTE


def test_low_aleatoric_coinflip_still_executes_under_gate():
    pred = arbiter.decide(_pred(p_t=0.5, u_aleatoric=0.05), aleatoric_max=0.18)
    assert pred.routing == Routing.EXECUTE
