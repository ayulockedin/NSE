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
    LatentPrediction,
    PlannerBranch,
    PruneReason,
    Routing,
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


# A comment-only patch that applies + compiles cleanly (so the GNN score, not the
# symbolic gate, decides the screen).
_COMMENT_PATCH = (
    "--- a/calc.py\n+++ b/calc.py\n@@ -1,4 +1,5 @@\n"
    ' """Tiny module under test for the NSE end-to-end smoke test."""\n'
    "+# reviewed\n \n \n def add(a, b):\n"
)


class TwoBranchPlanner:
    def __init__(self) -> None:
        self.good = str(uuid.uuid4())
        self.bad = str(uuid.uuid4())

    def plan(self, task, context):
        def branch(bid, strategy):
            return PlannerBranch(
                branch_id=bid, strategy=strategy, edited_files=["calc.py"],
                patch_preview=_COMMENT_PATCH, expected_complexity=0.1,
                planner_confidence=0.9,
            )
        return [branch(self.good, "good change"), branch(self.bad, "bad change")]


class StubEnsemble(LatentEnsemble):
    """Scores 'good' branches high and others near zero, so the GNN screen alone
    decides survival (independent of the patch text)."""

    def predict(self, branch, node_features=None, edge_index=None):
        p = 0.95 if "good" in branch.strategy else 0.02
        return LatentPrediction(
            branch_id=branch.branch_id, p_t_latent=p, r_long=0.1, u=0.0, per_head_p_t=[p]
        )


class RecordingSimulator:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def simulate(self, task, context, branches):
        self.seen = [b.branch_id for b in branches]
        return {
            b.branch_id: SimulatorPrediction(
                branch_id=b.branch_id, p_t_sim=0.9, explanation="ok"
            )
            for b in branches
        }


class UncertainEnsemble(LatentEnsemble):
    """High epistemic uncertainty -> the arbiter routes to INCREMENTAL_SANDBOX
    (gather evidence) rather than EXECUTE."""

    def predict(self, branch, node_features=None, edge_index=None):
        return LatentPrediction(
            branch_id=branch.branch_id, p_t_latent=0.6, r_long=0.1, u=0.3,
            per_head_p_t=[0.3, 0.9],
        )


def test_acquisition_resolves_uncertain_branch_via_evidence_run(db: DBClient):
    """Phase 10.2: with no EXECUTE finalist, spare sandbox budget is spent on the
    most-informative uncertain branch; a passing evidence run resolves the task."""
    orch = Orchestrator(
        db=db, planner=FakePlanner(), simulator=FakeSimulator(), critic=FakeCritic(),
        ensemble=UncertainEnsemble(), force_local_sandbox=True,
    )
    report = orch.run_task("review", TOY, ["calc.py"])

    assert report.predictions[0].routing == Routing.INCREMENTAL_SANDBOX  # not EXECUTE
    assert report.best_branch_id is not None        # resolved anyway, via evidence
    assert report.outcome_tests_passed == 1
    assert any("acquisition resolved" in n for n in report.notes)


def test_cascade_screens_hopeless_branch_before_llm(db: DBClient):
    """The GNN screen must keep a hopeless branch off the (expensive) LLM tier."""
    planner = TwoBranchPlanner()
    sim = RecordingSimulator()
    orch = Orchestrator(
        db=db, planner=planner, simulator=sim, critic=FakeCritic(),
        ensemble=StubEnsemble(), force_local_sandbox=True,
    )
    report = orch.run_task("edit calc", TOY, ["calc.py"])

    assert report.branches_generated == 2
    assert report.branches_judged == 1            # only the survivor reached tier 2
    assert sim.seen == [planner.good]             # the LLM never saw the screened-out branch
    assert report.best_branch_id == planner.good
    assert any("saved 1 LLM" in n for n in report.notes)

    bad = next(p for p in report.predictions if p.branch_id == planner.bad)
    assert bad.routing == Routing.PRUNE
    assert bad.prune_reason == PruneReason.LOW_SCORE
    assert bad.p_t_sim == 0.0                      # never judged -> no fabricated sim signal
