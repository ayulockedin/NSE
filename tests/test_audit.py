"""Tests for audit re-execution of pruned branches (false-negative hunting).

Uses the toy repo + the local (force_local) sandbox so the suite needs no Docker.
"""

import shutil
from pathlib import Path

from nse.config import ROOT
from nse.db.db_client import DBClient
from nse.orchestrator.audit import reexecute_pruned
from nse.orchestrator.schemas import PlannerBranch, PruneReason

TOY = ROOT / "nse" / "sandbox_repos" / "toy_repo"

GOOD = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
BROKEN = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"


def _db(tmp_path) -> DBClient:
    db = DBClient(tmp_path / "audit.sqlite3")
    db.init_schema()
    return db


def _seed(db, audit_dir, bid, reason, rewrite, *, retain=True):
    task_id = f"task-{bid}"
    db.insert_branch(
        task_id,
        PlannerBranch(
            branch_id=bid,
            strategy="s",
            edited_files=["calc.py"],
            full_file_rewrites={"calc.py": rewrite} if rewrite else None,
            expected_complexity=0.1,
            planner_confidence=0.5,
        ),
    )
    db.insert_pruned(bid, reason)
    if retain:
        dest = Path(audit_dir) / task_id / "repo"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(TOY, dest)


def _rows(db):
    return db.sample_pruned_for_audit(100)


def test_passing_low_score_prune_is_flagged_false_negative(tmp_path):
    db = _db(tmp_path)
    audit_dir = tmp_path / "snaps"
    _seed(db, audit_dir, "b-good", PruneReason.LOW_SCORE, GOOD)

    summary = reexecute_pruned(db, audit_dir, _rows(db), force_local=True)
    assert summary.reexecuted == 1
    assert summary.false_negatives == 1
    assert summary.false_negative_rate == 1.0
    # Recorded and reservoir drained.
    assert db.audit_false_negative_stats() == (1, 1)
    assert db.count_pruned_unsampled() == 0


def test_breaking_low_score_prune_is_not_false_negative(tmp_path):
    db = _db(tmp_path)
    audit_dir = tmp_path / "snaps"
    _seed(db, audit_dir, "b-broken", PruneReason.LOW_SCORE, BROKEN)

    summary = reexecute_pruned(db, audit_dir, _rows(db), force_local=True)
    assert summary.reexecuted == 1
    assert summary.false_negatives == 0
    n, fn = db.audit_false_negative_stats()
    assert (n, fn) == (1, 0)


def test_unsafe_patch_prune_is_never_reexecuted(tmp_path):
    db = _db(tmp_path)
    audit_dir = tmp_path / "snaps"
    _seed(db, audit_dir, "b-unsafe", PruneReason.UNSAFE_PATCH, GOOD)

    summary = reexecute_pruned(db, audit_dir, _rows(db), force_local=True)
    assert summary.reexecuted == 0
    assert "unsafe" in summary.results[0].note
    # Nothing executed -> no audit_results rows, but the row is still marked.
    assert db.audit_false_negative_stats() == (0, 0)
    assert db.count_pruned_unsampled() == 0


def test_missing_snapshot_is_skipped_gracefully(tmp_path):
    db = _db(tmp_path)
    audit_dir = tmp_path / "snaps"
    _seed(db, audit_dir, "b-nosnap", PruneReason.LOW_SCORE, GOOD, retain=False)

    summary = reexecute_pruned(db, audit_dir, _rows(db), force_local=True)
    assert summary.reexecuted == 0
    assert "snapshot" in summary.results[0].note
    assert db.count_pruned_unsampled() == 0  # still drained


def test_mixed_batch_false_negative_rate(tmp_path):
    db = _db(tmp_path)
    audit_dir = tmp_path / "snaps"
    _seed(db, audit_dir, "g1", PruneReason.LOW_SCORE, GOOD)
    _seed(db, audit_dir, "g2", PruneReason.LOW_SCORE, GOOD)
    _seed(db, audit_dir, "x1", PruneReason.LOW_SCORE, BROKEN)

    summary = reexecute_pruned(db, audit_dir, _rows(db), force_local=True)
    assert summary.reexecuted == 3
    assert summary.false_negatives == 2
    assert abs(summary.false_negative_rate - 2 / 3) < 1e-9
