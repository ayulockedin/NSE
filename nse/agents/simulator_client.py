"""Simulator agent — low-temperature analytic prior p_t^sim per branch."""

from __future__ import annotations

from typing import Any

from nse.agents.base_client import LLMClient
from nse.agents.prompts import SIMULATOR_SYSTEM, build_user_prompt
from nse.config import SETTINGS
from nse.orchestrator.schemas import PlannerBranch, SimulatorPrediction


class SimulatorClient(LLMClient):
    def simulate(
        self, task: str, context: dict[str, Any], branches: list[PlannerBranch]
    ) -> dict[str, SimulatorPrediction]:
        payload = {
            "context": context,
            "branches": [b.model_dump() for b in branches],
        }
        user = build_user_prompt(task, payload)
        preds = self.call_list(
            SIMULATOR_SYSTEM,
            user,
            SimulatorPrediction,
            temperature=SETTINGS.llm.simulator_temperature,
        )
        return {p.branch_id: p for p in preds}
