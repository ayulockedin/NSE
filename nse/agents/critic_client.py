"""Adversarial Critic agent — high-temperature red-team risk assessment."""

from __future__ import annotations

from typing import Any

from nse.agents.base_client import LLMClient
from nse.agents.prompts import CRITIC_SYSTEM, build_user_prompt
from nse.config import SETTINGS
from nse.orchestrator.schemas import CriticReport, PlannerBranch


class CriticClient(LLMClient):
    def critique(
        self, task: str, context: dict[str, Any], branches: list[PlannerBranch]
    ) -> dict[str, CriticReport]:
        payload = {
            "context": context,
            "branches": [b.model_dump() for b in branches],
        }
        user = build_user_prompt(task, payload)
        reports = self.call_list(
            CRITIC_SYSTEM,
            user,
            CriticReport,
            temperature=SETTINGS.llm.critic_temperature,
        )
        return {r.branch_id: r for r in reports}
