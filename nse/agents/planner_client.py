"""Planner agent — generates k candidate branches as strict JSON."""

from __future__ import annotations

import uuid
from typing import Any

from nse.agents.base_client import LLMClient
from nse.agents.prompts import PLANNER_SYSTEM, build_user_prompt
from nse.config import SETTINGS
from nse.orchestrator.schemas import PlannerBranch


class PlannerClient(LLMClient):
    def plan(self, task: str, context: dict[str, Any]) -> list[PlannerBranch]:
        user = build_user_prompt(task, context)
        branches = self.call_list(
            PLANNER_SYSTEM,
            user,
            PlannerBranch,
            temperature=SETTINGS.llm.planner_temperature,
        )
        # Real models often echo the example UUID, colliding branch_ids (and the
        # DB's INSERT OR REPLACE). Assign fresh ids — never trust the model's.
        for b in branches:
            b.branch_id = str(uuid.uuid4())
        return branches[: SETTINGS.hp.k]
