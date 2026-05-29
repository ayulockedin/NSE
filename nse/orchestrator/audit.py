"""Audit re-execution — close the false-negative loop.

The calibration loop samples pruned branches; this module *re-executes* them.
For each sampled pruned branch we replay its patch against the pristine per-task
snapshot the orchestrator retained, run the sandbox, and check whether the tests
actually pass. A branch pruned for ``low_score`` that nonetheless passes is a
**false negative** — NSE discarded a working patch, the safety-critical error
this audit exists to catch.

Safety: branches pruned as ``unsafe_patch`` are **never** re-executed — re-running
an exfiltration/secret-touching patch is exactly the action that prune prevented.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from nse.orchestrator.executor import run_sandbox
from nse.orchestrator.patcher import (
    PatchApplyError,
    PatchSafetyError,
    apply_patch_strict_then_fallback,
)
from nse.orchestrator.schemas import PlannerBranch, PruneReason

if TYPE_CHECKING:
    from nse.db.db_client import DBClient


@dataclass
class AuditResult:
    branch_id: str
    reason: str
    reexecuted: bool
    tests_passed: Optional[int]
    false_negative: bool
    note: str = ""


@dataclass
class AuditSummary:
    results: list[AuditResult] = field(default_factory=list)

    @property
    def sampled(self) -> int:
        return len(self.results)

    @property
    def reexecuted(self) -> int:
        return sum(1 for r in self.results if r.reexecuted)

    @property
    def false_negatives(self) -> int:
        return sum(1 for r in self.results if r.false_negative)

    @property
    def false_negative_rate(self) -> float | None:
        n = self.reexecuted
        return (self.false_negatives / n) if n else None

    def render(self) -> str:
        fnr = (
            "n/a"
            if self.false_negative_rate is None
            else f"{self.false_negative_rate:.3f}"
        )
        return (
            "-- Audit re-execution ------------------------------\n"
            f"  sampled        : {self.sampled}\n"
            f"  re-executed    : {self.reexecuted}\n"
            f"  false negatives: {self.false_negatives}  "
            "(pruned low_score but actually passed)\n"
            f"  FN rate        : {fnr}\n"
            "----------------------------------------------------"
        )


def _snapshot_dir(audit_dir: Path | str, task_id: str) -> Path:
    return Path(audit_dir) / task_id / "repo"


def reexecute_pruned(
    db: "DBClient",
    audit_dir: Path | str,
    rows: list[dict],
    force_local: bool = False,
) -> AuditSummary:
    """Replay each sampled pruned branch and record the outcome.

    ``rows`` are pruned-branch records (from ``db.sample_pruned_for_audit``).
    Every row is marked audited regardless of whether it could be re-run, so the
    reservoir drains. Returns an :class:`AuditSummary`.
    """
    audit_dir = Path(audit_dir)
    summary = AuditSummary()

    for row in rows:
        result = _audit_one(
            db, audit_dir, int(row["id"]), row["branch_id"], row["reason"], force_local
        )
        summary.results.append(result)
        db.mark_audited(int(row["id"]))

    return summary


def _audit_one(
    db: "DBClient",
    audit_dir: Path,
    pruned_id: int,
    branch_id: str,
    reason: str,
    force_local: bool,
) -> AuditResult:
    def skipped(note: str) -> AuditResult:
        return AuditResult(branch_id, reason, False, None, False, note)

    # Never re-run a patch that was pruned for being unsafe.
    if reason == PruneReason.UNSAFE_PATCH.value:
        return skipped("skipped: unsafe patch")

    rec = db.get_branch(branch_id)
    if rec is None:
        return skipped("skipped: branch not found")

    snapshot = _snapshot_dir(audit_dir, rec["task_id"])
    if not snapshot.exists():
        return skipped("skipped: no retained snapshot")

    branch = PlannerBranch.model_validate_json(rec["planner_json"])

    work = Path(tempfile.mkdtemp(prefix="nse_audit_"))
    repo = work / "repo"
    try:
        shutil.copytree(snapshot, repo)
        try:
            apply_patch_strict_then_fallback(
                repo,
                patch_text=branch.patch_preview,
                full_rewrites=branch.full_file_rewrites,
            )
        except (PatchApplyError, PatchSafetyError) as exc:
            return skipped(f"skipped: patch did not apply ({type(exc).__name__})")

        run = run_sandbox(
            repo,
            edited_files=branch.edited_files,
            full=True,
            force_local=force_local,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    tests_passed = int(run.tests_passed)
    false_negative = reason == PruneReason.LOW_SCORE.value and tests_passed == 1
    db.insert_audit_result(
        pruned_id=pruned_id,
        branch_id=branch_id,
        tests_passed=tests_passed,
        false_negative=int(false_negative),
        runtime=run.runtime,
    )
    return AuditResult(
        branch_id, reason, True, tests_passed, false_negative, run.mode
    )
