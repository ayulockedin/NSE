"""Tests for the real-defect resolution eval (Phase 11.4, % resolved).

Synthetic mined-style examples (no sandbox, no torch): each carries its label in
the unused depth slot ``features[5]`` so an oracle ensemble can read it, letting
us check the resolution accounting against a known-good and a known-blind model.
"""

import pytest

from nse.data.dataset import LabeledExample
from nse.eval.real_defect_eval import (
    evaluate_resolution,
    make_defect_tasks,
)
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator.schemas import LatentPrediction


def _ex(sha: str, label: int) -> LabeledExample:
    kind = "real_fix" if label == 1 else "real_regression"
    feats = [1.0, 2.0, 1.0, 6.0, 0.1, float(label)]  # depth slot carries the label
    return LabeledExample(
        feats, label, label == 0, kind, f"{sha} subject line", "calc.py",
        "git_mined", 0.0, node_features=[[0.0] * 8], edge_index=[[], []],
        r_long_target=0.0, weight=3.0,
    )


def _pair(sha: str) -> list[LabeledExample]:
    return [_ex(sha, 1), _ex(sha, 0)]


class _Oracle(LatentEnsemble):
    """Reads the true label from features[5] and predicts it exactly."""

    def predict_from_features(self, feats, branch_id="", node_features=None, edge_index=None):
        label = feats[5]
        return LatentPrediction(
            branch_id=branch_id, p_t_latent=label, r_long=0.0, u=0.0, per_head_p_t=[label]
        )


class _Blind(LatentEnsemble):
    """Always predicts failure -> everything scores below tau_prune."""

    def predict_from_features(self, feats, branch_id="", node_features=None, edge_index=None):
        return LatentPrediction(
            branch_id=branch_id, p_t_latent=0.0, r_long=0.0, u=0.0, per_head_p_t=[0.0]
        )


def test_make_defect_tasks_groups_by_sha():
    examples = _pair("aaaa1111") + _pair("bbbb2222")
    tasks = make_defect_tasks(examples)
    assert len(tasks) == 2
    assert all(len(t) == 2 for t in tasks)
    assert all(any(c.label == 1 for c in t) for t in tasks)


def test_make_defect_tasks_drops_groups_without_a_fix():
    # A lone regression (no fix) and a non-real example are both ignored.
    lone = [_ex("cccc3333", 0)]
    other = [LabeledExample([0.0] * 6, 1, False, "synthetic", "x", "m.py", "x", 0.0)]
    assert make_defect_tasks(lone + other) == []


def test_oracle_resolves_every_defect():
    tasks = make_defect_tasks(_pair("a1") + _pair("b2") + _pair("c3"))
    rep = evaluate_resolution(tasks, _Oracle())
    assert rep.n_defects == 3
    assert rep.resolved_rate == pytest.approx(1.0)
    assert rep.false_fix_rate == pytest.approx(0.0)
    assert rep.abstain_rate == pytest.approx(0.0)


def test_blind_model_abstains_and_resolves_nothing():
    tasks = make_defect_tasks(_pair("a1") + _pair("b2"))
    rep = evaluate_resolution(tasks, _Blind())
    assert rep.resolved_rate == pytest.approx(0.0)
    assert rep.abstain_rate == pytest.approx(1.0)
    assert rep.n_acted == 0
    assert rep.false_fix_rate is None  # nothing acted on


def test_heuristic_runs_and_reports_valid_rates():
    tasks = make_defect_tasks(_pair("a1") + _pair("b2"))
    rep = evaluate_resolution(tasks, LatentEnsemble())
    assert 0.0 <= rep.resolved_rate <= 1.0
    assert 0.0 <= rep.abstain_rate <= 1.0
    assert "% resolved" in rep.render()
