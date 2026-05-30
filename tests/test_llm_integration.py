"""Live-LLM integration tests (Phase 9).

Gated: they run only when NSE_LLM_BASE_URL is set AND the host is reachable
(e.g. ollama). Normal/CI runs skip immediately without any network call, so the
default suite stays offline and fast.

    set NSE_LLM_BASE_URL=http://127.0.0.1:11434/v1
    set NSE_LLM_MODEL=qwen2.5-coder:7b
    pytest tests/test_llm_integration.py
"""

import os

import pytest

from nse.agents.base_client import llm_available

_LLM_CONFIGURED = bool(os.environ.get("NSE_LLM_BASE_URL"))

pytestmark = pytest.mark.skipif(
    not (_LLM_CONFIGURED and llm_available()),
    reason="no live LLM configured (set NSE_LLM_BASE_URL to a reachable host)",
)

_CONTEXT = {
    "target_files": ["calc.py"],
    "nodes": [{"id": "calc.py::add", "signature": "add(a, b)"}],
    "source": {"calc.py": "def add(a, b):\n    return a + b\n"},
}
_TASK = "Add a one-line module docstring to calc.py without changing behavior."


def test_planner_returns_schema_valid_branches():
    from nse.agents.planner_client import PlannerClient

    planner = PlannerClient()
    branches = planner.plan(_TASK, _CONTEXT)
    assert len(branches) >= 1
    for b in branches:
        assert b.branch_id and isinstance(b.edited_files, list)
        assert 0.0 <= b.expected_complexity <= 1.0
    assert planner.tokens_used > 0  # token accounting works


def test_simulator_and_critic_return_schema():
    from nse.agents.critic_client import CriticClient
    from nse.agents.planner_client import PlannerClient
    from nse.agents.simulator_client import SimulatorClient

    branches = PlannerClient().plan(_TASK, _CONTEXT)
    sim = SimulatorClient().simulate(_TASK, _CONTEXT, branches)
    crit = CriticClient().critique(_TASK, _CONTEXT, branches)
    # Dicts keyed by branch_id; values already validated against the schemas.
    assert isinstance(sim, dict) and isinstance(crit, dict)
