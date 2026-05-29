"""Tests for the governance API + snapshot retention cap (Phase 13)."""

import os
import uuid

import pytest

from nse.api import GovernanceReport, NSEGovernor
from nse.config import ROOT
from nse.db.db_client import DBClient
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator.orchestrator import Orchestrator
from nse.orchestrator.schemas import (
    CriticReport,
    LatentPrediction,
    PlannerBranch,
    SimulatorPrediction,
)

TOY = ROOT / "nse" / "sandbox_repos" / "toy_repo"
_PATCH = (
    "--- a/calc.py\n+++ b/calc.py\n@@ -1,4 +1,5 @@\n"
    ' """Tiny module under test for the NSE end-to-end smoke test."""\n'
    "+# reviewed\n \n \n def add(a, b):\n"
)


class FakePlanner:
    def plan(self, task, context):
        return [PlannerBranch(
            branch_id=str(uuid.uuid4()), strategy="add a comment",
            edited_files=["calc.py"], patch_preview=_PATCH,
            expected_complexity=0.1, planner_confidence=0.9,
        )]


class FakeSimulator:
    def simulate(self, task, context, branches):
        return {b.branch_id: SimulatorPrediction(branch_id=b.branch_id, p_t_sim=0.9, explanation="ok")
                for b in branches}


class FakeCritic:
    def critique(self, task, context, branches):
        return {b.branch_id: CriticReport(branch_id=b.branch_id, r_critic=0.05) for b in branches}


class ConfidentEnsemble(LatentEnsemble):
    def predict(self, branch, node_features=None, edge_index=None):
        return LatentPrediction(
            branch_id=branch.branch_id, p_t_latent=0.92, r_long=0.1, u=0.0,
            per_head_p_t=[0.92],
        )


@pytest.fixture
def db(tmp_path) -> DBClient:
    return DBClient(tmp_path / "gov.sqlite3")


def test_governor_review_returns_report(db):
    gov = NSEGovernor(
        db=db, planner=FakePlanner(), simulator=FakeSimulator(), critic=FakeCritic(),
        ensemble=ConfidentEnsemble(), force_local_sandbox=True,
    )
    rep = gov.review("Add a review comment", TOY, ["calc.py"])

    assert isinstance(rep, GovernanceReport)
    assert rep.executed is True
    assert rep.resolved is True and rep.tests_passed == 1
    assert rep.branches_judged >= 1
    # Autonomy is *earned*: a single logged outcome is far below min_n, so the
    # governor asks for a human rather than auto-merging.
    assert rep.autonomy == "human_review"


def test_snapshot_retention_is_capped(db, tmp_path):
    orch = Orchestrator(
        db=db, planner=FakePlanner(), simulator=FakeSimulator(), critic=FakeCritic(),
        ensemble=ConfidentEnsemble(), force_local_sandbox=True,
    )
    # Create 5 task snapshots with increasing mtime, then keep only the newest 2.
    for i in range(5):
        d = orch.audit_dir / f"task-{i}" / "repo"
        d.mkdir(parents=True, exist_ok=True)
        (d / "x.py").write_text("x = 1\n", encoding="utf-8")
        os.utime(orch.audit_dir / f"task-{i}", (1000 + i, 1000 + i))

    orch._prune_old_snapshots(max_keep=2)
    remaining = sorted(d.name for d in orch.audit_dir.iterdir() if d.is_dir())
    assert remaining == ["task-3", "task-4"]   # oldest three pruned
