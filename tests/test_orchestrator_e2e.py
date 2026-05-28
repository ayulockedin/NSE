"""End-to-end pipeline test with injected fake agents (no network/GPU).

Exercises: memory graph -> planner -> symbolic gate -> simulator -> latent
ensemble -> critic -> arbiter -> local sandbox -> DB logging.
"""

import uuid
from pathlib import Path

import pytest

from nse.config import ROOT
from nse.db.db_client import DBClient
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator.orchestrator import Orchestrator
from nse.orchestrator.schemas import (
    CriticReport,
    PlannerBranch,
    SimulatorPrediction,
)

TOY = ROOT / "nse" / "sandbox_repos" / "toy_repo"


class FakePlanner:
    def __init__(self) -> None:
        self.bid = str(uuid.uuid4())

    def plan(self, task, context):
        patch = (
            "--- a/calc.py\n+++ b/calc.py\n@@ -1,4 +1,5 @@\n"
            ' """Tiny module under test for the NSE end-to-end smoke test."""\n'
            "+# reviewed\n \n \n def add(a, b):\n"
        )
        return [
            PlannerBranch(
                branch_id=self.bid,
                strategy="add a comment, no behavioural change",
                edited_files=["calc.py"],
                patch_preview=patch,
                expected_complexity=0.1,
                planner_confidence=0.9,
            )
        ]


class FakeSimulator:
    def simulate(self, task, context, branches):
        return {
            b.branch_id: SimulatorPrediction(
                branch_id=b.branch_id, p_t_sim=0.9, explanation="trivial"
            )
            for b in branches
        }


class FakeCritic:
    def critique(self, task, context, branches):
        return {
            b.branch_id: CriticReport(
                branch_id=b.branch_id, r_critic=0.05, identified_failures=[],
                attack_confidence=0.1,
            )
            for b in branches
        }


@pytest.fixture
def db(tmp_path: Path) -> DBClient:
    return DBClient(tmp_path / "test.sqlite3")


def test_pipeline_runs_and_logs(db: DBClient):
    orch = Orchestrator(
        db=db,
        planner=FakePlanner(),
        simulator=FakeSimulator(),
        critic=FakeCritic(),
        ensemble=LatentEnsemble(),
        force_local_sandbox=True,
    )
    report = orch.run_task("Add a review comment to calc.py", TOY, ["calc.py"])

    assert report.branches_generated == 1
    assert report.best_branch_id is not None
    assert report.outcome_tests_passed == 1       # toy tests still pass
    assert report.sandbox_mode == "local_unsafe"
    assert report.wall_clock_s >= 0.0

    # DB was populated.
    joined = db.predictions_with_outcomes()
    assert len(joined) == 1
    assert joined[0]["tests_passed"] == 1


def test_unsafe_patch_is_pruned(db: DBClient):
    class EnvPlanner(FakePlanner):
        def plan(self, task, context):
            return [
                PlannerBranch(
                    branch_id=str(uuid.uuid4()),
                    strategy="write secrets",
                    edited_files=[".env"],
                    full_file_rewrites={".env": "SECRET=1\n"},
                    expected_complexity=0.1,
                    planner_confidence=0.9,
                )
            ]

    orch = Orchestrator(
        db=db, planner=EnvPlanner(), simulator=FakeSimulator(),
        critic=FakeCritic(), ensemble=LatentEnsemble(),
        force_local_sandbox=True,
    )
    report = orch.run_task("exfiltrate", TOY, ["calc.py"])
    assert report.best_branch_id is None
    assert report.predictions[0].prune_reason is not None
