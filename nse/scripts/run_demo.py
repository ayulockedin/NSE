"""End-to-end demo: drive the full orchestrator with a real (or mock) LLM.

Real model (ollama) — no mock needed:
    set NSE_LLM_BASE_URL=http://127.0.0.1:11434/v1
    set NSE_LLM_MODEL=qwen2.5-coder:7b
    python -m nse.scripts.run_demo

Or against the mock: start it first (python -m nse.scripts.vllm_mock).
"""

from __future__ import annotations

import json

from nse.config import ROOT
from nse.orchestrator.orchestrator import Orchestrator

TOY = ROOT / "nse" / "sandbox_repos" / "toy_repo"


def main() -> None:
    orch = Orchestrator(force_local_sandbox=True)
    report = orch.run_task(
        "Add a validation comment to the add() function in calc.py",
        TOY,
        ["calc.py"],
    )
    print(
        json.dumps(
            {
                "task_id": report.task_id,
                "branches_generated": report.branches_generated,
                "best_branch_id": report.best_branch_id,
                "outcome_compiled": report.outcome_compiled,
                "outcome_tests_passed": report.outcome_tests_passed,
                "sandbox_mode": report.sandbox_mode,
                "wall_clock_s": report.wall_clock_s,
                "routings": [
                    {"branch": p.branch_id[:8], "route": p.routing.value if p.routing else None,
                     "score": round(p.score, 3) if p.score is not None else None,
                     "p_t": round(p.p_t, 3), "u": round(p.u, 4)}
                    for p in report.predictions
                ],
                "notes": report.notes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
