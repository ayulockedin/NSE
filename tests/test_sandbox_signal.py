"""Tests for the richer sandbox signal (Phase 8.2). Uses the local runner."""

import shutil

from nse.config import ROOT
from nse.orchestrator.executor import _parse_pytest_summary, run_sandbox

TOY = ROOT / "nse" / "sandbox_repos" / "toy_repo"
BROKEN = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"


def test_parse_summary_passing():
    assert _parse_pytest_summary("5 passed in 0.12s") == (5, 0, [])


def test_parse_summary_with_failure():
    out = "1 failed, 4 passed in 0.2s\nFAILED test_calc.py::test_add - assert"
    n_passed, n_failed, failed = _parse_pytest_summary(out)
    assert (n_passed, n_failed) == (4, 1)
    assert failed == ["test_calc.py::test_add"]


def test_passing_run_reports_counts():
    run = run_sandbox(TOY, full=True, force_local=True)
    assert run.tests_passed is True
    assert run.n_passed >= 2
    assert run.n_failed == 0
    assert run.failed_tests == []


def test_failing_run_captures_failed_test(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(TOY, repo)
    (repo / "calc.py").write_text(BROKEN, encoding="utf-8")
    run = run_sandbox(repo, full=True, force_local=True)
    assert run.tests_passed is False
    assert run.n_failed >= 1
    assert any("test_add" in t for t in run.failed_tests)


def test_pytest_targets_pins_the_designated_test_set():
    # Pin to one explicit test file; this overrides full-suite selection and is
    # how the miner scopes around env-broken peripheral tests.
    run = run_sandbox(TOY, pytest_targets=["test_calc.py"], force_local=True)
    assert run.tests_passed is True
    assert run.n_passed >= 1
