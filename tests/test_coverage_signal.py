"""Tests for the coverage-as-uncertainty signal (Phase 7.1)."""

import pytest

from nse.config import ROOT
from nse.tools.coverage_signal import (
    FileCoverage,
    changed_lines,
    changed_lines_from_diff,
    coverage_available,
    coverage_fraction,
    coverage_uncertainty,
    measure_line_coverage,
)

FULL = 0.25  # full_scale used in assertions


def _cov(executed, executable):
    return {"m.py": FileCoverage(executed=set(executed), executable=set(executable))}


# ──────────────────────────── diff parsing ──────────────────────────────


def test_diff_replacement_uses_old_line_numbers():
    diff = (
        "--- a/m.py\n+++ b/m.py\n@@ -4,2 +4,2 @@\n"
        " def add(a, b):\n-    return a + b\n+    return a - b\n"
    )
    # The replaced (removed) line in the OLD file is line 5.
    assert changed_lines_from_diff(diff) == {"m.py": {5}}


def test_diff_pure_insertion_falls_back_to_context():
    diff = "--- a/m.py\n+++ b/m.py\n@@ -4,1 +4,2 @@\n def add(a, b):\n+    x = 1\n"
    # No removals -> use the hunk's context old-line (4) as locality proxy.
    assert changed_lines_from_diff(diff) == {"m.py": {4}}


def test_changed_lines_handles_full_rewrite():
    out = changed_lines(None, {"m.py": "a\nb\nc"}, {"m.py": 3})
    assert out == {"m.py": {1, 2, 3}}


def test_changed_lines_combines_diff_and_rewrite():
    diff = "--- a/a.py\n+++ b/a.py\n@@ -2,1 +2,1 @@\n-x = 1\n+x = 2\n"
    out = changed_lines(diff, {"b.py": "p\nq"}, {"b.py": 2})
    assert out == {"a.py": {2}, "b.py": {1, 2}}


# ─────────────────────────── uncertainty map ────────────────────────────


def test_fully_covered_change_has_zero_uncertainty():
    assert coverage_uncertainty(_cov({4, 5}, {4, 5}), {"m.py": {5}}, FULL) == 0.0


def test_uncovered_change_gets_full_scale():
    # line 5 is executable but not executed -> fully under-tested.
    assert coverage_uncertainty(_cov({4}, {4, 5}), {"m.py": {5}}, FULL) == FULL


def test_partial_coverage_scales_linearly():
    # 1 of 2 executable changed lines covered -> half the full scale.
    cov = _cov({5}, {5, 6})
    assert coverage_uncertainty(cov, {"m.py": {5, 6}}, FULL) == pytest.approx(FULL / 2)


def test_non_executable_changed_lines_are_excluded():
    # Changing only comment/blank lines (not in executable set) yields no signal,
    # so a comment insertion never looks falsely under-tested.
    assert coverage_uncertainty(_cov({4, 5}, {4, 5}), {"m.py": {1, 2, 3}}, FULL) == 0.0


def test_missing_coverage_map_is_noop():
    assert coverage_uncertainty(None, {"m.py": {5}}, FULL) == 0.0
    assert coverage_fraction(None, {"m.py": {5}}) is None


def test_empty_change_is_noop():
    assert coverage_uncertainty(_cov({1}, {1}), {}, FULL) == 0.0


# ──────────────────────── real measurement (opt) ─────────────────────────


@pytest.mark.skipif(not coverage_available(), reason="coverage.py not installed")
def test_measure_line_coverage_on_toy_repo():
    toy = ROOT / "nse" / "sandbox_repos" / "toy_repo"
    cov = measure_line_coverage(toy)
    assert cov is not None
    # add()/sub() bodies are exercised by test_calc.py -> non-empty coverage.
    assert cov["calc.py"].executed
