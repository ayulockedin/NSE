"""System / user prompt templates for the agent triad (blueprint section 5).

Context is placed first so it lands in the shared vLLM prefix (prefix caching);
the role-specific instruction is the suffix that varies per agent.
"""

from __future__ import annotations

import json
from typing import Any

from nse.config import SETTINGS

PLANNER_SYSTEM = (
    "You are PLANNER. Given the structured JSON context, output exactly "
    f"k={SETTINGS.hp.k} distinct, realistic implementation strategies as a "
    "JSON array of PlannerBranch objects. Each object must have: branch_id "
    "(uuid), strategy (1-2 sentences), edited_files (list), patch_preview "
    "(unified diff) or full_file_rewrites ({path: text}), assumptions (list), "
    "expected_complexity (0..1), planner_confidence (0..1). Aim for diversity "
    "across approaches. Strict JSON only — no prose."
)

SIMULATOR_SYSTEM = (
    "You are SIMULATOR. For each branch, estimate p_t_sim (0..1) — the "
    "probability the change passes tests — and give a concise logic summary. "
    "Output a JSON array of {branch_id, p_t_sim, explanation}. Strict JSON only."
)

CRITIC_SYSTEM = (
    "You are CRITIC (red-team). For each branch, find a minimal failing "
    "input/scenario, concurrency or security holes. Output a JSON array of "
    "{branch_id, r_critic(0..1), identified_failures:[...], "
    "attack_confidence(0..1)}. Strict JSON only."
)

REPAIR_SYSTEM = (
    "repair_and_output_json: The previous output was not valid JSON matching "
    "the required schema. Re-emit ONLY the corrected JSON. No commentary."
)


def build_user_prompt(task: str, context: dict[str, Any]) -> str:
    return (
        f"CONTEXT:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"TASK:\n{task}"
    )
