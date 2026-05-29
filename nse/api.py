"""Governance-substrate API (Phase 13) — the stable facade any agent plugs into.

NSE's positioning is *the trust layer*, not another coding agent. :class:`NSEGovernor`
is the pluggable entry point: hand it a task + repo and it runs the full cost-tiered
cascade (GNN screen → LLM judge → sandbox), then returns a structured
:class:`GovernanceReport` — the decision, the verified outcome, the
**competence-aware autonomy level** for the change, and the cost — decoupled from
orchestrator internals so callers depend only on this surface.

Example::

    gov = NSEGovernor(force_local_sandbox=False)            # real Docker + LLM via env
    report = gov.review("Fix the off-by-one in slice()", repo_path, ["seq.py"])
    if report.autonomy == "auto_merge" and report.resolved:
        merge(...)
    elif report.executed:
        request_human_review(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nse.db.db_client import DBClient
from nse.orchestrator.autonomy import (
    AutonomyLevel,
    autonomy_for,
    compute_competence,
    default_segment_fn,
)
from nse.orchestrator.orchestrator import Orchestrator, TaskReport


@dataclass
class GovernanceReport:
    """The public decision record for one governed change."""

    task_id: str
    executed: bool                   # a branch was executed (or evidence-resolved)
    resolved: bool                   # executed AND its tests passed
    autonomy: str                    # AutonomyLevel for the change's segment
    branches_screened: int
    branches_judged: int
    cost_units: float
    tests_passed: Optional[int]
    sandbox_mode: Optional[str]
    notes: list[str] = field(default_factory=list)


class NSEGovernor:
    """Stable facade over the orchestrator. Extra kwargs (``planner``, ``simulator``,
    ``critic``, ``ensemble``) pass through to :class:`Orchestrator` for injection."""

    def __init__(
        self,
        db: Optional[DBClient] = None,
        force_local_sandbox: bool = False,
        **orchestrator_kwargs,
    ) -> None:
        self.orch = Orchestrator(
            db=db, force_local_sandbox=force_local_sandbox, **orchestrator_kwargs
        )
        self.db = self.orch.db

    def review(
        self,
        task: str,
        repo_path: Path | str,
        target_files: Optional[list[str]] = None,
    ) -> GovernanceReport:
        """Run the cascade on ``task`` and return a governance decision."""
        return self._to_report(self.orch.run_task(task, repo_path, target_files))

    def autonomy_level(self, segment_file: str) -> AutonomyLevel:
        """The earned autonomy level for a segment (file), from logged history."""
        return autonomy_for(segment_file, compute_competence(self.db.labeled_predictions()))

    def _to_report(self, r: TaskReport) -> GovernanceReport:
        autonomy = AutonomyLevel.HUMAN_REVIEW
        if r.best_branch_id is not None:
            row = self.db.get_branch(r.best_branch_id)
            if row:
                autonomy = self.autonomy_level(default_segment_fn(row["planner_json"]))
        return GovernanceReport(
            task_id=r.task_id,
            executed=r.best_branch_id is not None,
            resolved=r.outcome_tests_passed == 1,
            autonomy=autonomy.value,
            branches_screened=r.branches_screened,
            branches_judged=r.branches_judged,
            cost_units=r.cost_units,
            tests_passed=r.outcome_tests_passed,
            sandbox_mode=r.sandbox_mode,
            notes=list(r.notes),
        )
