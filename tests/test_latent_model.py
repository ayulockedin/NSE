from nse.config import SETTINGS
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator.schemas import PlannerBranch


def _branch(complexity: float) -> PlannerBranch:
    return PlannerBranch(
        branch_id="b",
        strategy="s",
        edited_files=["x.py"],
        patch_preview="--- a/x.py\n+++ b/x.py\n@@\n+pass\n",
        expected_complexity=complexity,
        planner_confidence=0.7,
    )


def test_prediction_bounds():
    pred = LatentEnsemble().predict(_branch(0.5))
    assert 0.0 <= pred.p_t_latent <= 1.0
    assert 0.0 <= pred.r_long <= 1.0
    assert pred.u >= 0.0
    assert len(pred.per_head_p_t) == SETTINGS.hp.M


def test_uncertainty_grows_with_complexity():
    low = LatentEnsemble().predict(_branch(0.1))
    high = LatentEnsemble().predict(_branch(0.9))
    assert high.u >= low.u
    # harder change -> lower predicted pass probability
    assert high.p_t_latent <= low.p_t_latent


def test_uncertainty_decomposition_matches_formula():
    pred = LatentEnsemble().predict(_branch(0.3))
    expected = sum(p * (1 - p) for p in pred.per_head_p_t) / len(pred.per_head_p_t)
    assert abs(pred.u_aleatoric - expected) < 1e-9  # mean per-head Bernoulli variance
    assert 0.0 <= pred.u_aleatoric <= 0.25          # bounded by the Bernoulli max
