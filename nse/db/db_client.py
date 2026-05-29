"""Thin SQLite client for predictions / outcomes / audit logging.

SQLite is the MVP store (the blueprint allows SQLite or Postgres). The API is
deliberately small and synchronous; swap the connection factory for psycopg to
move to Postgres without touching call sites.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from nse.config import DB_PATH, SCHEMA_PATH
from nse.orchestrator.schemas import (
    BranchPrediction,
    Outcome,
    PlannerBranch,
    PruneReason,
)


class DBClient:
    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")

    # ── lifecycle ──────────────────────────────────────────────────────
    def init_schema(self, schema_path: Path | str = SCHEMA_PATH) -> None:
        sql = Path(schema_path).read_text(encoding="utf-8")
        self._conn.executescript(sql)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DBClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── writes ─────────────────────────────────────────────────────────
    def insert_branch(self, task_id: str, branch: PlannerBranch) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO branches (id, task_id, planner_json) VALUES (?, ?, ?)",
            (branch.branch_id, task_id, branch.model_dump_json()),
        )
        self._conn.commit()

    def insert_prediction(self, pred: BranchPrediction) -> None:
        self._conn.execute(
            """INSERT INTO predictions
               (branch_id, p_t, u, r_critic, r_long, c_planner, p_c, score, routing)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred.branch_id,
                pred.p_t,
                pred.u,
                pred.r_critic,
                pred.r_long,
                pred.c_planner,
                pred.p_c,
                pred.score,
                pred.routing.value if pred.routing else None,
            ),
        )
        self._conn.commit()

    def insert_outcome(self, outcome: Outcome) -> None:
        self._conn.execute(
            """INSERT INTO outcomes (branch_id, compiled, tests_passed, logs, runtime)
               VALUES (?, ?, ?, ?, ?)""",
            (
                outcome.branch_id,
                outcome.compiled,
                outcome.tests_passed,
                outcome.logs,
                outcome.runtime,
            ),
        )
        self._conn.commit()

    def insert_pruned(self, branch_id: str, reason: PruneReason) -> None:
        self._conn.execute(
            "INSERT INTO pruned_branches (branch_id, reason) VALUES (?, ?)",
            (branch_id, reason.value),
        )
        self._conn.commit()

    def insert_calibration(
        self, ece: float, brier: float, action: str = ""
    ) -> None:
        self._conn.execute(
            "INSERT INTO calibrations (ece, brier, action) VALUES (?, ?, ?)",
            (ece, brier, action),
        )
        self._conn.commit()

    # ── reads (audit / calibration) ────────────────────────────────────
    def sample_pruned_for_audit(self, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT * FROM pruned_branches
               WHERE sampled_for_audit = 0
               ORDER BY RANDOM() LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_audited(self, pruned_id: int) -> None:
        self._conn.execute(
            "UPDATE pruned_branches SET sampled_for_audit = 1 WHERE id = ?",
            (pruned_id,),
        )
        self._conn.commit()

    def predictions_with_outcomes(self) -> list[dict[str, Any]]:
        """Joined view for calibration (predicted p_t vs observed tests_passed)."""
        rows = self._conn.execute(
            """SELECT p.branch_id, p.p_t, o.tests_passed
               FROM predictions p JOIN outcomes o ON p.branch_id = o.branch_id"""
        ).fetchall()
        return [dict(r) for r in rows]

    def count_tasks(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT task_id) AS n FROM branches"
        ).fetchone()
        return int(row["n"])

    def count_pruned_unsampled(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM pruned_branches WHERE sampled_for_audit = 0"
        ).fetchone()
        return int(row["n"])
