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

from nse.agents.base_client import AgentError
from nse.agents.critic_client import CriticClient
from nse.agents.planner_client import PlannerClient
from nse.agents.simulator_client import SimulatorClient
from nse.config import SETTINGS
from nse.db.db_client import DBClient
from nse.memory_graph.context import get_prompt_context
from nse.memory_graph.graph_db import MemoryGraph
from nse.models.calibration_loop import (
    detect_drift,
    load_conformal_threshold,
    load_recalibrator,
    run_calibration,
)
from nse.models.cpg_features import build_cpg_features, changed_function_names
from nse.models.latent_model import LatentEnsemble, load_ensemble
from nse.orchestrator import arbiter
from nse.orchestrator.audit import reexecute_pruned
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
from nse.tools.coverage_signal import (
    FileCoverage,
    changed_lines,
    coverage_uncertainty,
    measure_line_coverage,
)
from nse.tools.property_oracle import golden_source, run_property_oracle
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
        # Auto-load trained latent weights when present; the heuristic fallback
        # keeps the pipeline running before any training has happened.
        self.ensemble = ensemble or load_ensemble(strict=False)
        # Persisted recalibrator from the calibration loop (None until the first
        # recalibration); applied to aggregated p_t in the arbiter.
        self.recalibrator = load_recalibrator()
        # Conformal EXECUTE gate threshold (None until enough calibration data).
        self.conformal_threshold = load_conformal_threshold()
        self.force_local_sandbox = force_local_sandbox
        # Pristine per-task snapshots retained here so the audit loop can replay
        # pruned branches. Kept next to the DB (git-ignored) -> tmp DB in tests.
        self.audit_dir = self.db.db_path.parent / "audit_snapshots"

    # ── per-branch scoring ─────────────────────────────────────────────
    def _score_branch(
        self,
        snapshot: Path,
        branch: PlannerBranch,
        sim_p_t: float,
        r_critic: float,
        mg: Optional[MemoryGraph] = None,
        coverage_map: Optional[dict[str, FileCoverage]] = None,
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
            # Identify the edited function(s) while the patched tree still
            # exists, so we can center the CPG subgraph exactly as the dataset
            # builder did (per-function granularity, no train/serve skew).
            edited_fns = (
                changed_function_names(snapshot, repo, branch.edited_files)
                if mg is not None and branch.edited_files
                else []
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

        # Same CPG-lite featurizer the dataset builder used.
        node_features, edge_index = (
            build_cpg_features(mg, branch.edited_files, center_functions=edited_fns)
            if mg is not None and branch.edited_files
            else (None, None)
        )
        latent = self.ensemble.predict(
            branch, node_features=node_features, edge_index=edge_index
        )
        coverage_u = self._coverage_uncertainty(branch, snapshot, coverage_map)
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
        arbiter.decide(
            pred,
            recalibrator=self.recalibrator,
            coverage_u=coverage_u,
            conformal_threshold=self.conformal_threshold,
        )
        # Phase 7.3: for a branch sent to gather evidence (under-tested / no
        # conformal guarantee), synthesize a property oracle. A new crash is
        # unambiguous breakage -> prune; a clean run leaves it gathering evidence.
        if pred.routing == Routing.INCREMENTAL_SANDBOX and edited_fns:
            self._apply_oracle_evidence(pred, branch, snapshot, edited_fns[0])
        return pred

    def _apply_oracle_evidence(
        self,
        pred: BranchPrediction,
        branch: PlannerBranch,
        snapshot: Path,
        func_name: str,
    ) -> None:
        golden = golden_source(snapshot, branch.edited_files[0], func_name)
        if golden is None:
            return
        run = run_property_oracle(
            snapshot, branch, func_name, golden,
            force_local=self.force_local_sandbox,
        )
        if run is not None and not run.tests_passed:
            pred.routing = Routing.PRUNE
            pred.prune_reason = PruneReason.ORACLE_CRASH

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

        # Reset per-task LLM token meters across the agent triad (no-op for fakes).
        for agent in (self.planner, self.simulator, self.critic):
            reset = getattr(agent, "reset_tokens", None)
            if callable(reset):
                reset()

        def check_budget() -> None:
            if time.time() - start > SETTINGS.budgets.max_wall_clock_per_task_s:
                raise BudgetExceeded("wall-clock budget exceeded")
            tokens = sum(
                getattr(a, "tokens_used", 0)
                for a in (self.planner, self.simulator, self.critic)
            )
            if tokens > SETTINGS.budgets.max_total_llm_tokens_per_task:
                raise BudgetExceeded(f"LLM token budget exceeded ({tokens} tokens)")

        # 1. Memory graph + context.
        mg = MemoryGraph(repo_path).build_graph()
        mg.build_test_map()
        target_files = target_files or [
            n["file"] for _, n in mg.g.nodes(data=True) if n.get("kind") == "module"
        ][:1]
        context = get_prompt_context(mg, target_files, depth=1)
        # Coverage of the pristine (trusted) repo, measured once per task. Feeds
        # coverage-as-uncertainty in the arbiter (Phase 7.1). None = no signal.
        coverage_map = measure_line_coverage(repo_path)

        # 2. Planner. A real model that emits unrepairable JSON shouldn't crash
        # the task — degrade to "no branches" and finish cleanly.
        try:
            branches = self.planner.plan(task, context)
        except AgentError as exc:
            report.notes.append(f"planner failed: {exc}")
            return self._finalize(report, start)
        report.branches_generated = len(branches)
        for b in branches:
            self.db.insert_branch(task_id, b)

        # Retain a pristine snapshot so the audit loop can later replay any
        # branch this task prunes.
        self._retain_snapshot(task_id, repo_path)

        # 3. Simulator + Critic (batched per task; share the prefix). A failure in
        # either degrades to neutral signals (the arbiter uses defaults), never a
        # crash.
        try:
            sim = self.simulator.simulate(task, context, branches)
        except AgentError as exc:
            sim = {}
            report.notes.append(f"simulator failed: {exc}")
        try:
            crit = self.critic.critique(task, context, branches)
        except AgentError as exc:
            crit = {}
            report.notes.append(f"critic failed: {exc}")

        # 4. Score every branch.
        snapshot = Path(tempfile.mkdtemp(prefix="nse_snap_")) / "repo"
        shutil.copytree(repo_path, snapshot)
        try:
            for b in branches:
                check_budget()
                sim_p_t = sim[b.branch_id].p_t_sim if b.branch_id in sim else 0.5
                r_critic = crit[b.branch_id].r_critic if b.branch_id in crit else 0.0
                pred = self._score_branch(
                    snapshot, b, sim_p_t, r_critic, mg=mg, coverage_map=coverage_map
                )
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
    def _coverage_uncertainty(
        branch: PlannerBranch,
        snapshot: Path,
        coverage_map: Optional[dict[str, FileCoverage]],
    ) -> float:
        """Uncertainty that this branch's edited lines are under-tested.

        For full-file rewrites the whole pristine file counts, so we read its
        length from the (still-pristine) ``snapshot``."""
        if coverage_map is None or not branch.edited_files:
            return 0.0
        counts: dict[str, int] = {}
        for f in branch.full_file_rewrites or {}:
            fp = snapshot / f
            if fp.exists():
                counts[f] = len(fp.read_text(encoding="utf-8").splitlines())
        changed = changed_lines(branch.patch_preview, branch.full_file_rewrites, counts)
        return coverage_uncertainty(coverage_map, changed, SETTINGS.hp.coverage_u_full)

    def _retain_snapshot(self, task_id: str, repo_path: Path) -> None:
        """Copy the pristine repo aside so the audit loop can replay pruned
        branches against the exact state they were scored on. Heavy/regenerable
        dirs are skipped; existing snapshots are left as-is."""
        dest = self.audit_dir / task_id / "repo"
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            repo_path,
            dest,
            ignore=shutil.ignore_patterns(
                "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"
            ),
        )

    def _finalize(self, report: TaskReport, start: float) -> TaskReport:
        report.wall_clock_s = round(time.time() - start, 3)
        self._maybe_run_controls(report)
        return report

    def _maybe_run_controls(self, report: TaskReport) -> None:
        """Phase-6 control loop: every N tasks recalibrate on logged history and
        sample the pruned reservoir for audit. Cadence from ``Hyperparams``."""
        hp = SETTINGS.hp
        n_tasks = self.db.count_tasks()
        if hp.calibration_retrain_every and n_tasks % hp.calibration_retrain_every == 0:
            result = run_calibration(self.db)
            report.notes.append(
                f"calibration: {result.action} (ece={result.ece:.3f}, n={result.n}, "
                f"conformal_tau={result.conformal_threshold})"
            )
            if result.recalibrated:
                self.recalibrator = load_recalibrator()
            # The conformal threshold can change even without recalibration.
            self.conformal_threshold = load_conformal_threshold()
            # Drift: if recalibration can't keep up, flag a full retrain and
            # persist the recommendation for an offline job to pick up.
            drift = detect_drift(self.db)
            if drift.drifted:
                report.notes.append(
                    f"DRIFT: ece={drift.ece:.3f} > {SETTINGS.hp.drift_ece_threshold} "
                    f"— {drift.recommendation}"
                )
                self.db.insert_calibration(drift.ece, drift.brier, drift.recommendation)
        if hp.audit_every_n_tasks and n_tasks % hp.audit_every_n_tasks == 0:
            total = self.db.count_pruned_unsampled()
            if total and hp.audit_sampling_percent > 0:
                limit = max(1, round(hp.audit_sampling_percent * total))
                rows = self.db.sample_pruned_for_audit(limit)
                summary = reexecute_pruned(
                    self.db, self.audit_dir, rows,
                    force_local=self.force_local_sandbox,
                )
                report.notes.append(
                    f"audit: re-ran {summary.reexecuted}/{summary.sampled} pruned, "
                    f"{summary.false_negatives} false-negative(s)"
                )
