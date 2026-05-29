"""Tests for the cost-tiered cascade (Phase 10.1).

Stubs stand in for the GNN screen / LLM judge / sandbox verify so the control
flow + cost accounting are tested offline. The key invariant: the GNN screen
keeps hopeless branches off the (expensive) LLM and sandbox tiers.
"""

import pytest

from nse.orchestrator.cascade import run_cascade
from nse.orchestrator.cost import CostModel
from nse.orchestrator.schemas import LatentPrediction

# Per-candidate behaviour: GNN latent, (sim p_t, tokens), sandbox outcome.
# 'mid' scores well enough on the GNN to clear the screen (r_long penalty kept at
# 0), but the LLM judge (sim=0.0) drags its aggregate score below tau at tier 2.
LAT = {
    "good": LatentPrediction(branch_id="", p_t_latent=0.92, r_long=0.1, u=0.0),
    "mid": LatentPrediction(branch_id="", p_t_latent=0.6, r_long=0.0, u=0.0),
    "hopeless": LatentPrediction(branch_id="", p_t_latent=0.05, r_long=0.1, u=0.0),
}
SIM = {"good": (0.9, 200), "mid": (0.0, 200), "hopeless": (0.5, 200)}
VERIFY = {"good": True, "mid": False, "hopeless": False}


def _stubs():
    calls = {"screen": [], "judge": [], "verify": []}

    def screen(c):
        calls["screen"].append(c)
        return LAT[c]

    def judge(c):
        calls["judge"].append(c)
        return SIM[c]

    def verify(c):
        calls["verify"].append(c)
        return VERIFY[c]

    return calls, screen, judge, verify


def test_screen_keeps_hopeless_branch_off_the_expensive_tiers():
    calls, screen, judge, verify = _stubs()
    cm = CostModel(gnn_cost=0.001, llm_cost_per_1k_tokens=0.05, sandbox_cost_per_run=1.0)
    res = run_cascade(["good", "mid", "hopeless"], screen, judge, verify, cost_model=cm)

    # GNN screened all 3; the hopeless one was pruned before any LLM spend.
    assert calls["screen"] == ["good", "mid", "hopeless"]
    assert "hopeless" not in calls["judge"]
    assert res.llm_calls_saved == 1

    # 'mid' survives the screen but the LLM downgrades it -> pruned at tier 2.
    assert set(calls["judge"]) == {"good", "mid"}
    assert res.survivors == ["good"]

    # Exactly the finalist is sandboxed, once.
    assert res.finalist == "good"
    assert res.verified is True
    assert calls["verify"] == ["good"]
    assert res.sandbox_runs_saved == 2

    # Cost ledger: gnn on all 3, llm on 2 survivors, sandbox once.
    assert res.ledger.gnn == pytest.approx(0.003)
    assert res.ledger.llm == pytest.approx(2 * 0.05 * 0.2)
    assert res.ledger.sandbox == pytest.approx(1.0)


def test_tier_counts():
    _calls, screen, judge, verify = _stubs()
    res = run_cascade(["good", "mid", "hopeless"], screen, judge, verify)
    names = [(t.name, t.n_in, t.n_out) for t in res.tiers]
    assert names == [
        ("gnn_screen", 3, 2),
        ("llm_judge", 2, 1),
        ("sandbox_verify", 1, 1),
    ]


def test_all_pruned_at_screen_runs_no_llm_or_sandbox():
    calls, screen, judge, verify = _stubs()
    res = run_cascade(["hopeless"], screen, judge, verify)
    assert res.finalist is None
    assert res.verified is None
    assert calls["judge"] == [] and calls["verify"] == []
    assert res.llm_calls_saved == 1
    assert res.sandbox_runs_saved == 1
    assert res.ledger.llm == 0.0 and res.ledger.sandbox == 0.0


def test_render_is_stringable():
    _calls, screen, judge, verify = _stubs()
    res = run_cascade(["good", "hopeless"], screen, judge, verify)
    out = res.render()
    assert "Cascade" in out and "finalist" in out
