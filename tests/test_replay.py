"""Tests for the replay flywheel (Phase 8.1)."""

import shutil

import pytest

from nse.config import ROOT
from nse.data.dataset import LabeledExample
from nse.data.replay import example_weight, harvest_training_examples
from nse.db.db_client import DBClient
from nse.models.latent_model import _TORCH
from nse.orchestrator.schemas import BranchPrediction, Outcome, PlannerBranch, PruneReason

TOY = ROOT / "nse" / "sandbox_repos" / "toy_repo"
REWRITE = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"


def _db(tmp_path) -> DBClient:
    db = DBClient(tmp_path / "replay.sqlite3")
    db.init_schema()
    return db


def _retain(audit_dir, task_id):
    dest = audit_dir / task_id / "repo"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TOY, dest)


def _branch(bid):
    return PlannerBranch(
        branch_id=bid, strategy="s", edited_files=["calc.py"],
        full_file_rewrites={"calc.py": REWRITE},
        expected_complexity=0.1, planner_confidence=0.5,
    )


def test_example_weight_grows_with_error():
    correct = example_weight(0.9, 1)
    wrong = example_weight(0.1, 1)        # confident but actually passed
    assert wrong > correct
    assert example_weight(1.0, 1) == pytest.approx(1.0)   # perfect -> no up-weight
    assert example_weight(0.0, 1) == pytest.approx(5.0)   # max error -> 1 + scale


def test_harvest_reconstructs_executed_outcome(tmp_path):
    db = _db(tmp_path)
    audit_dir = tmp_path / "snaps"
    _retain(audit_dir, "task-1")
    db.insert_branch("task-1", _branch("b1"))
    db.insert_prediction(BranchPrediction(branch_id="b1", p_c=1, p_t=0.2))
    db.insert_outcome(Outcome(branch_id="b1", compiled=1, tests_passed=1))

    harvested = harvest_training_examples(db, audit_dir)
    assert len(harvested) == 1
    ex = harvested[0]
    assert ex.label == 1
    assert len(ex.features) == 6
    assert ex.node_features  # real CPG graph rebuilt from the snapshot
    # predicted 0.2 but actually passed -> strongly up-weighted.
    assert ex.weight == pytest.approx(example_weight(0.2, 1))


def test_harvest_includes_audit_false_negatives(tmp_path):
    db = _db(tmp_path)
    audit_dir = tmp_path / "snaps"
    _retain(audit_dir, "task-2")
    db.insert_branch("task-2", _branch("b2"))
    db.insert_prediction(BranchPrediction(branch_id="b2", p_c=1, p_t=0.1))
    db.insert_pruned("b2", PruneReason.LOW_SCORE)
    pruned_id = db.sample_pruned_for_audit(1)[0]["id"]
    db.insert_audit_result(pruned_id, "b2", tests_passed=1, false_negative=1, runtime=0.0)

    harvested = harvest_training_examples(db, audit_dir)
    assert len(harvested) == 1 and harvested[0].label == 1


def test_harvest_skips_when_snapshot_missing(tmp_path):
    db = _db(tmp_path)
    audit_dir = tmp_path / "snaps"  # nothing retained
    db.insert_branch("task-3", _branch("b3"))
    db.insert_prediction(BranchPrediction(branch_id="b3", p_c=1, p_t=0.5))
    db.insert_outcome(Outcome(branch_id="b3", compiled=1, tests_passed=1))
    assert harvest_training_examples(db, audit_dir) == []


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_weighted_training_runs_with_nonunit_weights():
    from nse.models.latent_model import LatentEnsemble
    from nse.models.train import train

    pos = [
        LabeledExample([1.0, 2, 0, 6, 0.1, 0], 1, False, "k", "d", "m.py",
                       "x", 0.0, weight=5.0)
        for _ in range(12)
    ]
    neg = [
        LabeledExample([1.0, 1, 1, 6, 0.1, 0], 0, True, "k", "d", "m.py",
                       "x", 0.0, weight=1.0)
        for _ in range(12)
    ]
    model = train(pos + neg, epochs=40, seed=0)
    ens = LatentEnsemble(model=model)
    p = ens.predict_from_features(pos[0].features)
    assert 0.0 <= p.p_t_latent <= 1.0
