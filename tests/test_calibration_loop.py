"""Tests for the Phase-6 calibration + audit control loop."""

import pytest

from nse.db.db_client import DBClient
from nse.models.calibrate import (
    _SKLEARN,
    Recalibrator,
    compute_ece,
    fit_isotonic_recalibrator,
)
from nse.models.calibration_loop import load_recalibrator, run_audit, run_calibration
from nse.orchestrator import arbiter
from nse.orchestrator.schemas import (
    BranchPrediction,
    Outcome,
    PlannerBranch,
    PruneReason,
    Routing,
)

# Isotonic recalibration needs scikit-learn; skip those cases when it's absent.
requires_sklearn = pytest.mark.skipif(not _SKLEARN, reason="scikit-learn not installed")


# ──────────────────────────── fixtures / seeding ────────────────────────


def _db(tmp_path) -> DBClient:
    db = DBClient(tmp_path / "cal.sqlite3")
    db.init_schema()
    return db


def _seed_pairs(db: DBClient, pairs: list[tuple[float, int]]) -> None:
    for i, (p, y) in enumerate(pairs):
        bid = f"b{i}"
        db.insert_branch(
            f"task-{i}",
            PlannerBranch(
                branch_id=bid,
                strategy="s",
                edited_files=["f.py"],
                expected_complexity=0.1,
                planner_confidence=0.5,
            ),
        )
        db.insert_prediction(BranchPrediction(branch_id=bid, p_c=1, p_t=p))
        db.insert_outcome(
            Outcome(branch_id=bid, compiled=1, tests_passed=y)
        )


def _miscalibrated() -> list[tuple[float, int]]:
    """Overconfident: predicted v but true pass-rate v**2."""
    pairs: list[tuple[float, int]] = []
    for v in (0.2, 0.4, 0.6, 0.8):
        k = 25
        n_pass = round(v * v * k)
        pairs += [(v, 1 if j < n_pass else 0) for j in range(k)]
    return pairs


def _well_calibrated() -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    for v in (0.2, 0.5, 0.8):
        k = 25
        n_pass = round(v * k)
        pairs += [(v, 1 if j < n_pass else 0) for j in range(k)]
    return pairs


# ──────────────────────────── recalibrator ──────────────────────────────


def test_recalibrator_json_roundtrip():
    r = Recalibrator(xs=[0.0, 0.5, 1.0], ys=[0.0, 0.25, 1.0])
    r2 = Recalibrator.from_json(r.to_json())
    assert r2.xs == r.xs and r2.ys == r.ys
    assert abs(r2(0.5) - 0.25) < 1e-9
    assert abs(r2(0.25) - 0.125) < 1e-9  # linear interpolation between knots


@requires_sklearn
def test_isotonic_recalibrator_reduces_ece():
    pairs = _miscalibrated()
    probs = [p for p, _ in pairs]
    labels = [y for _, y in pairs]
    recal = fit_isotonic_recalibrator(probs, labels)
    before = compute_ece(probs, labels)
    after = compute_ece([recal(p) for p in probs], labels)
    assert after < before
    assert after < 0.05


# ──────────────────────────── run_calibration ───────────────────────────


@requires_sklearn
def test_run_calibration_recalibrates_when_miscalibrated(tmp_path):
    db = _db(tmp_path)
    _seed_pairs(db, _miscalibrated())
    path = tmp_path / "recal.json"
    result = run_calibration(db, persist_path=path)
    assert result.action == "recalibrated_isotonic"
    assert result.recalibrated is True
    assert result.ece > 0.05
    assert path.exists()
    # the calibration event is logged for traceability
    n_cal = db._conn.execute("SELECT COUNT(*) AS n FROM calibrations").fetchone()["n"]
    assert n_cal == 1


def test_run_calibration_ok_when_well_calibrated(tmp_path):
    db = _db(tmp_path)
    _seed_pairs(db, _well_calibrated())
    path = tmp_path / "recal.json"
    result = run_calibration(db, persist_path=path)
    assert result.action == "ok"
    assert result.recalibrated is False
    assert not path.exists()  # nothing persisted when already calibrated


def test_run_calibration_skips_on_insufficient_data(tmp_path):
    db = _db(tmp_path)
    _seed_pairs(db, [(0.8, 1), (0.2, 0)])  # only 2 samples
    result = run_calibration(db, min_samples=20)
    assert result.action == "skipped_insufficient_data"
    assert result.recalibrated is False


@requires_sklearn
def test_load_recalibrator_roundtrips_through_disk(tmp_path):
    db = _db(tmp_path)
    _seed_pairs(db, _miscalibrated())
    path = tmp_path / "recal.json"
    run_calibration(db, persist_path=path)
    loaded = load_recalibrator(path)
    assert loaded is not None
    assert 0.0 <= loaded(0.8) <= 1.0


def test_load_recalibrator_absent_returns_none(tmp_path):
    assert load_recalibrator(tmp_path / "missing.json") is None


# ─────────────────────────────── run_audit ──────────────────────────────


def test_run_audit_samples_and_marks_reservoir(tmp_path):
    db = _db(tmp_path)
    for i in range(20):
        db.insert_pruned(f"p{i}", PruneReason.LOW_SCORE)
    assert db.count_pruned_unsampled() == 20
    sampled = run_audit(db, sampling_percent=0.5)
    assert len(sampled) == 10
    assert db.count_pruned_unsampled() == 10  # the sampled ones are marked
    # second pass samples from what remains
    again = run_audit(db, sampling_percent=0.5)
    assert len(again) == 5


def test_run_audit_empty_reservoir(tmp_path):
    db = _db(tmp_path)
    assert run_audit(db) == []


# ───────────────────── arbiter applies recalibrator ─────────────────────


def test_arbiter_recalibrator_changes_decision_but_keeps_raw_p_t():
    pred = BranchPrediction(
        branch_id="b", p_c=1, p_t=0.8, u=0.0, c_planner=1.0, r_critic=0.0, r_long=0.0
    )
    base = arbiter.decide(BranchPrediction(**pred.model_dump()))
    # Collapsing p_t lowers the score (stable invariant) while the logged
    # pred.p_t stays the raw 0.8. The exact routing depends on hyperparameters,
    # so we only assert it moves out of EXECUTE into a conservative decision.
    recal_pred = arbiter.decide(
        BranchPrediction(**pred.model_dump()), recalibrator=lambda p: 0.05
    )
    assert recal_pred.p_t == 0.8
    assert recal_pred.score < base.score
    assert base.routing == Routing.EXECUTE
    assert recal_pred.routing != Routing.EXECUTE
