"""Tests for competence-aware autonomy (Phase 12.2). Offline, synthetic rows."""

import json

from nse.orchestrator.autonomy import (
    AutonomyLevel,
    autonomy_for,
    compute_competence,
    default_segment_fn,
)


def _rows(file: str, p_t: float, label: int, k: int) -> list[dict]:
    pj = json.dumps({"edited_files": [file]})
    return [{"p_t": p_t, "tests_passed": label, "planner_json": pj} for _ in range(k)]


def test_default_segment_fn():
    assert default_segment_fn(json.dumps({"edited_files": ["pkg/x.py"]})) == "pkg/x.py"
    assert default_segment_fn(json.dumps({"edited_files": []})) == "unknown"
    assert default_segment_fn("not json") == "unknown"


def test_competence_levels_by_segment():
    rows = (
        # a.py: confident + correct, plenty of evidence -> AUTO_MERGE
        _rows("a.py", 0.95, 1, 28) + _rows("a.py", 0.05, 0, 2)
        # b.py: overconfident, often fails -> miscalibrated/low-precision -> ASSISTED
        + _rows("b.py", 0.9, 0, 20) + _rows("b.py", 0.9, 1, 10)
        # c.py: too little evidence -> HUMAN_REVIEW
        + _rows("c.py", 0.8, 1, 5)
    )
    comp = compute_competence(rows)

    assert comp["a.py"].n == 30
    assert comp["a.py"].level() == AutonomyLevel.AUTO_MERGE
    assert comp["a.py"].precision == 1.0

    assert comp["b.py"].level() == AutonomyLevel.ASSISTED
    assert comp["b.py"].precision < 0.5      # predicts pass, mostly fails

    assert comp["c.py"].level() == AutonomyLevel.HUMAN_REVIEW   # n=5 < min_n


def test_autonomy_for_unknown_segment_defaults_to_review():
    comp = compute_competence(_rows("a.py", 0.95, 1, 30))
    assert autonomy_for("a.py", comp) == AutonomyLevel.AUTO_MERGE
    assert autonomy_for("never-seen.py", comp) == AutonomyLevel.HUMAN_REVIEW
