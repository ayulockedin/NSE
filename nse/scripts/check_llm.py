"""Validate the configured LLM end-to-end through the agent triad.

Pings the configured OpenAI-compatible host, then runs Planner / Simulator /
Critic once and confirms each returns schema-valid JSON (with token usage).

Point at ollama first:
    set NSE_LLM_BASE_URL=http://127.0.0.1:11434/v1
    set NSE_LLM_MODEL=qwen2.5-coder:7b
    python -m nse.scripts.check_llm
"""

from __future__ import annotations

import sys

from nse.agents.base_client import AgentError, llm_available
from nse.agents.critic_client import CriticClient
from nse.agents.planner_client import PlannerClient
from nse.agents.simulator_client import SimulatorClient
from nse.config import SETTINGS

_CONTEXT = {
    "target_files": ["calc.py"],
    "functions": ["add(a, b)", "sub(a, b)"],
    "summary": "a tiny calculator module with add and sub",
}
_TASK = "Add a one-line module docstring to calc.py without changing behavior."


def main() -> int:
    print(f"LLM host : {SETTINGS.llm.base_url}")
    print(f"model    : {SETTINGS.llm.model}")
    if not llm_available():
        print("  [x] host not reachable - set NSE_LLM_BASE_URL / NSE_LLM_MODEL.")
        return 1
    print("  [ok] reachable\n")

    try:
        planner = PlannerClient()
        branches = planner.plan(_TASK, _CONTEXT)
        print(f"PLANNER  : {len(branches)} branch(es), {planner.tokens_used} tokens")
        for b in branches:
            print(f"   - [{b.expected_complexity:.2f}] {b.strategy[:80]}")
        if not branches:
            print("  [x] planner returned no valid branches")
            return 1

        sim = SimulatorClient()
        preds = sim.simulate(_TASK, _CONTEXT, branches)
        print(f"SIMULATOR: {len(preds)} pred(s), {sim.tokens_used} tokens")

        crit = CriticClient()
        reports = crit.critique(_TASK, _CONTEXT, branches)
        print(f"CRITIC   : {len(reports)} report(s), {crit.tokens_used} tokens")
    except AgentError as exc:
        print(f"  [x] agent could not produce schema-valid JSON: {exc}")
        return 1

    total = planner.tokens_used + sim.tokens_used + crit.tokens_used
    print(f"\n[ok] agent triad returned schema-valid JSON from the real model "
          f"({total} tokens total).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
