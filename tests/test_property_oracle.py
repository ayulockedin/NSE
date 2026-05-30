"""Tests for the property-based / metamorphic oracle (Phase 7.3)."""

import ast

from nse.config import ROOT
from nse.orchestrator.schemas import PlannerBranch
from nse.tools.property_oracle import (
    build_harness,
    differential_check,
    generate_inputs,
    golden_source,
    run_property_oracle,
)

TOY = ROOT / "nse" / "sandbox_repos" / "toy_repo"

GOLDEN = "def f(a, b):\n    return a + b\n"
BENIGN = "def f(a, b):\n    _x = 0\n    return a + b\n"
DIVERGENT = "def f(a, b):\n    return a * 2\n"          # differs, never crashes
CRASHING = "def f(a, b):\n    return a + b + ([][0])\n"  # always crashes


def _branch(rewrite: str) -> PlannerBranch:
    return PlannerBranch(
        branch_id="b",
        strategy="s",
        edited_files=["calc.py"],
        full_file_rewrites={"calc.py": rewrite},
        expected_complexity=0.1,
        planner_confidence=0.5,
    )


# ──────────────────────────── input generation ──────────────────────────


def test_generate_inputs_deterministic_and_arity():
    assert generate_inputs(0) == [()]
    a = generate_inputs(2, n=20, seed=1)
    b = generate_inputs(2, n=20, seed=1)
    assert a == b and all(len(t) == 2 for t in a)
    assert generate_inputs(2, n=20, seed=2) != a  # seed changes the draw


# ─────────────────────────── differential check ─────────────────────────


def test_benign_change_passes_clean():
    r = differential_check(GOLDEN, BENIGN, "f")
    assert r.ok and r.new_crashes == 0 and r.divergences == 0
    assert r.comparable > 0  # the test actually exercised the functions


def test_pure_divergence_is_not_a_crash():
    r = differential_check(GOLDEN, DIVERGENT, "f")
    assert r.ok                      # no new crashes -> gate passes
    assert r.new_crashes == 0
    assert r.divergences > 0         # but behavior change is recorded


def test_crash_introducing_change_is_flagged():
    r = differential_check(GOLDEN, CRASHING, "f")
    assert not r.ok and r.new_crashes > 0


def test_missing_function_is_inconclusive_not_failing():
    r = differential_check(GOLDEN, "def other(): pass\n", "f")
    assert r.ok and "not found" in r.note


# ───────────────────────────── harness build ────────────────────────────


def test_build_harness_is_valid_importable_python():
    src = build_harness(GOLDEN, "f", "calc", generate_inputs(2, n=5))
    ast.parse(src)  # must be syntactically valid
    assert "import calc as _m" in src
    assert "def test_no_new_crashes()" in src


def test_golden_source_extracts_function():
    src = golden_source(TOY, "calc.py", "add")
    assert src is not None and "def add" in src


# ──────────────────── secure (sandboxed) runner ─────────────────────────


def test_run_property_oracle_detects_crash():
    gsrc = golden_source(TOY, "calc.py", "add")
    rewrite = "def add(a, b):\n    return a + b + ([][0])\n\n\ndef sub(a, b):\n    return a - b\n"
    run = run_property_oracle(TOY, _branch(rewrite), "add", gsrc, force_local=True)
    assert run is not None and run.tests_passed is False  # crash caught


def test_run_property_oracle_passes_benign():
    gsrc = golden_source(TOY, "calc.py", "add")
    rewrite = "def add(a, b):\n    _x = 0\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
    run = run_property_oracle(TOY, _branch(rewrite), "add", gsrc, force_local=True)
    assert run is not None and run.tests_passed is True


def test_run_property_oracle_returns_none_when_unappliable():
    gsrc = golden_source(TOY, "calc.py", "add")
    bad = PlannerBranch(
        branch_id="b", strategy="s", edited_files=["calc.py"],
        patch_preview="this is not a valid diff", expected_complexity=0.1,
        planner_confidence=0.5,
    )
    assert run_property_oracle(TOY, bad, "add", gsrc, force_local=True) is None


# ─────────────────────── orchestrator evidence hook ─────────────────────


def test_orchestrator_oracle_evidence_prunes_on_crash(tmp_path):
    from nse.db.db_client import DBClient
    from nse.orchestrator.orchestrator import Orchestrator
    from nse.orchestrator.schemas import BranchPrediction, PruneReason, Routing

    orch = Orchestrator(db=DBClient(tmp_path / "o.sqlite3"), force_local_sandbox=True)
    crash = "def add(a, b):\n    return a + b + ([][0])\n\n\ndef sub(a, b):\n    return a - b\n"
    pred = BranchPrediction(
        branch_id="b", p_c=1, p_t=0.8, routing=Routing.INCREMENTAL_SANDBOX
    )
    orch._apply_oracle_evidence(pred, _branch(crash), TOY, "add")
    assert pred.routing == Routing.PRUNE
    assert pred.prune_reason == PruneReason.ORACLE_CRASH


def test_orchestrator_oracle_evidence_keeps_benign(tmp_path):
    from nse.db.db_client import DBClient
    from nse.orchestrator.orchestrator import Orchestrator
    from nse.orchestrator.schemas import BranchPrediction, Routing

    orch = Orchestrator(db=DBClient(tmp_path / "o.sqlite3"), force_local_sandbox=True)
    benign = "def add(a, b):\n    _x = 0\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
    pred = BranchPrediction(
        branch_id="b", p_c=1, p_t=0.8, routing=Routing.INCREMENTAL_SANDBOX
    )
    orch._apply_oracle_evidence(pred, _branch(benign), TOY, "add")
    assert pred.routing == Routing.INCREMENTAL_SANDBOX  # no new crash -> unchanged
