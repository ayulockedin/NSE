"""Tests for the Phase-6 calibration + audit control loop."""

import pytest

from nse.db.db_client import DBClient
from nse.models.calibrate import (
    _SKLEARN,
    CONFORMAL_NEVER,
    Recalibrator,
    binomial_upper_bound,
    compute_ece,
    fit_conformal_threshold,
    fit_isotonic_recalibrator,
)
from nse.models.calibration_loop import (
    detect_drift,
    load_conformal_threshold,
    load_recalibrator,
    run_audit,
    run_calibration,
)
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
    result = run_calibration(db, persist_path=path, conformal_path=tmp_path / "conf.json")
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
    result = run_calibration(db, persist_path=path, conformal_path=tmp_path / "conf.json")
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
    run_calibration(db, persist_path=path, conformal_path=tmp_path / "conf.json")
    loaded = load_recalibrator(path)
    assert loaded is not None
    assert 0.0 <= loaded(0.8) <= 1.0


def test_load_recalibrator_absent_returns_none(tmp_path):
    assert load_recalibrator(tmp_path / "missing.json") is None


# ───────────────────────── conformal gate (7.2) ─────────────────────────


def test_binomial_upper_bound_is_a_valid_bound():
    assert binomial_upper_bound(0, 0, 0.1) == 1.0          # no data -> trivial
    assert binomial_upper_bound(50, 50, 0.1) == 1.0        # all failed
    assert binomial_upper_bound(0, 50, 0.1) <= 0.05        # 0/50 -> tight
    assert binomial_upper_bound(5, 50, 0.1) > 0.05         # 10% -> exceeds alpha
    # More trials at the same rate -> tighter bound.
    assert binomial_upper_bound(2, 100, 0.1) < binomial_upper_bound(1, 20, 0.1)


def test_conformal_threshold_controls_false_execute_rate():
    # high p_t -> pass, low p_t -> fail; the gate must pick a τ whose executed
    # set's empirical false rate respects alpha.
    probs = [0.9 if i % 2 else 0.2 for i in range(100)]
    labels = [1 if p > 0.5 else 0 for p in probs]
    tau = fit_conformal_threshold(probs, labels, alpha=0.05, delta=0.1)
    assert tau is not None and tau < CONFORMAL_NEVER
    executed = [(p, y) for p, y in zip(probs, labels) if p >= tau]
    false_rate = sum(1 for _, y in executed if y == 0) / len(executed)
    assert false_rate <= 0.05


def test_conformal_threshold_conservative_when_unachievable():
    # Both classes present, but the only confidence tier fails ~50% -> no τ can
    # guarantee the false-execute rate -> execute nothing (conservative sentinel).
    labels = [i % 2 for i in range(40)]  # 20 pass / 20 fail, all at p_t=0.9
    assert fit_conformal_threshold([0.9] * 40, labels, 0.05, 0.1) == CONFORMAL_NEVER


def test_conformal_threshold_none_on_sparse_data():
    assert fit_conformal_threshold([0.9, 0.2], [1, 0], 0.05, 0.1) is None


def test_run_calibration_persists_conformal_threshold(tmp_path):
    db = _db(tmp_path)
    # Well-calibrated with a clean, fully-passing top tier (50 @ 0.99 all pass)
    # so the gate is findable without recalibration (no sklearn needed).
    pairs = [(0.99, 1)] * 50 + [(0.4, 1)] * 20 + [(0.4, 0)] * 30
    _seed_pairs(db, pairs)
    conf_path = tmp_path / "conf.json"
    result = run_calibration(
        db, persist_path=tmp_path / "recal.json", conformal_path=conf_path
    )
    assert result.conformal_threshold == 0.99
    assert conf_path.exists()
    assert load_conformal_threshold(conf_path) == 0.99


def test_load_conformal_threshold_absent_returns_none(tmp_path):
    assert load_conformal_threshold(tmp_path / "missing.json") is None


# ──────────────────────────── drift (8.3) ───────────────────────────────


def test_detect_drift_flags_badly_miscalibrated_model(tmp_path):
    db = _db(tmp_path)
    _seed_pairs(db, _miscalibrated())  # ECE ~0.2, above the 0.15 drift threshold
    report = detect_drift(db)
    assert report.drifted is True
    assert report.recommendation == "full_retrain_recommended"
    assert report.ece > 0.15


def test_detect_drift_ok_when_calibrated(tmp_path):
    db = _db(tmp_path)
    _seed_pairs(db, _well_calibrated())
    report = detect_drift(db)
    assert report.drifted is False
    assert report.recommendation == "ok"


def test_detect_drift_insufficient_data(tmp_path):
    db = _db(tmp_path)
    _seed_pairs(db, [(0.9, 1), (0.2, 0)])
    report = detect_drift(db)
    assert report.drifted is False
    assert report.recommendation == "insufficient_data"


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
