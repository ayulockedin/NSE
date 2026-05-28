"""Orchestrator — end-to-end NSE task flow (blueprint section 8.1).

    task -> snapshot -> memory graph -> Planner (k branches)
         -> per branch: apply patch -> symbolic gate -> simulator
                        -> latent ensemble -> critic -> predictions
         -> Arbiter (score + route) -> sandbox-execute best -> log outcome

Strict per-task budgets are enforced throughout (section 8.2).
"""

from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nse.agents.critic_client import CriticClient
from nse.agents.planner_client import PlannerClient
from nse.agents.simulator_client import SimulatorClient
from nse.config import SETTINGS
from nse.db.db_client import DBClient
from nse.memory_graph.context import get_prompt_context
from nse.memory_graph.graph_db import MemoryGraph
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator import arbiter
from nse.orchestrator.executor import run_sandbox, to_outcome
from nse.orchestrator.patcher import (
    PatchApplyError,
    PatchSafetyError,
    apply_patch_strict_then_fallback,
)
from nse.orchestrator.schemas import (
    BranchPrediction,
    PlannerBranch,
    PruneReason,
    Routing,
)
from nse.tools.static_checks import symbolic_check


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class TaskReport:
    task_id: str
    branches_generated: int = 0
    predictions: list[BranchPrediction] = field(default_factory=list)
    best_branch_id: Optional[str] = None
    outcome_compiled: Optional[int] = None
    outcome_tests_passed: Optional[int] = None
    sandbox_mode: Optional[str] = None
    wall_clock_s: float = 0.0
    notes: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        db: Optional[DBClient] = None,
        planner: Optional[PlannerClient] = None,
        simulator: Optional[SimulatorClient] = None,
        critic: Optional[CriticClient] = None,
        ensemble: Optional[LatentEnsemble] = None,
        force_local_sandbox: bool = False,
    ) -> None:
        self.db = db or DBClient()
        self.db.init_schema()
        self.planner = planner or PlannerClient()
        self.simulator = simulator or SimulatorClient()
        self.critic = critic or CriticClient()
        self.ensemble = ensemble or LatentEnsemble()
        self.force_local_sandbox = force_local_sandbox

    # ── per-branch scoring ─────────────────────────────────────────────
    def _score_branch(
        self,
        snapshot: Path,
        branch: PlannerBranch,
        sim_p_t: float,
        r_critic: float,
    ) -> BranchPrediction:
        # Apply patch into an isolated copy for the symbolic gate.
        work = Path(tempfile.mkdtemp(prefix="nse_score_"))
        repo = work / "repo"
        try:
            shutil.copytree(snapshot, repo)
            try:
                apply_patch_strict_then_fallback(
                    repo,
                    patch_text=branch.patch_preview,
                    full_rewrites=branch.full_file_rewrites,
                )
            except PatchSafetyError:
                return self._gate_fail(branch, PruneReason.UNSAFE_PATCH)
            except PatchApplyError:
                return self._gate_fail(branch, PruneReason.SYMBOLIC_FAIL)

            sym = symbolic_check(repo)
        finally:
            shutil.rmtree(work, ignore_errors=True)

        latent = self.ensemble.predict(branch)
        p_t = arbiter.aggregate_p_t(latent.p_t_latent, sim_p_t)
        pred = BranchPrediction(
            branch_id=branch.branch_id,
            p_c=sym.p_c,
            p_t_sim=sim_p_t,
            p_t_latent=latent.p_t_latent,
            p_t=p_t,
            u=latent.u,
            r_critic=r_critic,
            r_long=latent.r_long,
            c_planner=branch.planner_confidence,
        )
        return arbiter.decide(pred)

    @staticmethod
    def _gate_fail(branch: PlannerBranch, reason: PruneReason) -> BranchPrediction:
        return BranchPrediction(
            branch_id=branch.branch_id,
            p_c=0,
            c_planner=branch.planner_confidence,
            routing=Routing.PRUNE,
            prune_reason=reason,
        )

    # ── main entrypoint ────────────────────────────────────────────────
    def run_task(
        self,
        task: str,
        repo_path: Path | str,
        target_files: Optional[list[str]] = None,
    ) -> TaskReport:
        start = time.time()
        task_id = str(uuid.uuid4())
        repo_path = Path(repo_path)
        report = TaskReport(task_id=task_id)

        def check_budget() -> None:
            if time.time() - start > SETTINGS.budgets.max_wall_clock_per_task_s:
                raise BudgetExceeded("wall-clock budget exceeded")

        # 1. Memory graph + context.
        mg = MemoryGraph(repo_path).build_graph()
        mg.build_test_map()
        target_files = target_files or [
            n["file"] for _, n in mg.g.nodes(data=True) if n.get("kind") == "module"
        ][:1]
        context = get_prompt_context(mg, target_files, depth=1)

        # 2. Planner.
        branches = self.planner.plan(task, context)
        report.branches_generated = len(branches)
        for b in branches:
            self.db.insert_branch(task_id, b)

        # 3. Simulator + Critic (batched per task; share the prefix).
        sim = self.simulator.simulate(task, context, branches)
        crit = self.critic.critique(task, context, branches)

        # 4. Score every branch.
        snapshot = Path(tempfile.mkdtemp(prefix="nse_snap_")) / "repo"
        shutil.copytree(repo_path, snapshot)
        try:
            for b in branches:
                check_budget()
                sim_p_t = sim[b.branch_id].p_t_sim if b.branch_id in sim else 0.5
                r_critic = crit[b.branch_id].r_critic if b.branch_id in crit else 0.0
                pred = self._score_branch(snapshot, b, sim_p_t, r_critic)
                report.predictions.append(pred)
                self.db.insert_prediction(pred)
                if pred.routing == Routing.PRUNE and pred.prune_reason:
                    self.db.insert_pruned(b.branch_id, pred.prune_reason)

            # 5. Arbiter selection.
            best = arbiter.select_best(report.predictions)
            if best is None:
                # Nothing executable; surface uncertainty-routed branches.
                routed = [
                    p for p in report.predictions
                    if p.routing == Routing.INCREMENTAL_SANDBOX
                ]
                report.notes.append(
                    f"no branch routed to EXECUTE; "
                    f"{len(routed)} awaiting incremental evidence"
                )
                return self._finalize(report, start)

            report.best_branch_id = best.branch_id
            best_branch = next(b for b in branches if b.branch_id == best.branch_id)

            # 6. Apply best patch to snapshot and sandbox-execute.
            check_budget()
            apply_patch_strict_then_fallback(
                snapshot,
                patch_text=best_branch.patch_preview,
                full_rewrites=best_branch.full_file_rewrites,
            )
            run = run_sandbox(
                snapshot,
                edited_files=best_branch.edited_files,
                tests_map=mg.tests_map,
                force_local=self.force_local_sandbox,
            )
            outcome = to_outcome(best.branch_id, run)
            self.db.insert_outcome(outcome)
            report.outcome_compiled = outcome.compiled
            report.outcome_tests_passed = outcome.tests_passed
            report.sandbox_mode = run.mode
        finally:
            shutil.rmtree(snapshot.parent, ignore_errors=True)

        return self._finalize(report, start)

    @staticmethod
    def _finalize(report: TaskReport, start: float) -> TaskReport:
        report.wall_clock_s = round(time.time() - start, 3)
        return report
