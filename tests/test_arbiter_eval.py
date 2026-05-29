"""Tests for arbiter-level (selection) evaluation."""

from nse.data.dataset import LabeledExample
from nse.eval.arbiter_eval import Task, evaluate_arbiter, make_tasks
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator.schemas import LatentPrediction


def _ex(label: int, file: str = "a.py", complexity: float = 0.0) -> LabeledExample:
    # features[0] carries the label so the oracle stub can read it; features[4]
    # is expected_complexity (what the real heuristic reads).
    return LabeledExample(
        features=[float(label), 0.0, 0.0, 0.0, complexity, 0.0],
        label=label,
        assumed_breaking=bool(1 - label),
        kind="k",
        description="d",
        file=file,
        sandbox_mode="local_unsafe",
        runtime=0.0,
    )


class _OracleEnsemble:
    """Stub whose p_t_latent equals features[0] (the label) — a perfect ranker."""

    def predict_from_features(self, feats, node_features=None, edge_index=None):
        p = feats[0]
        return LatentPrediction(
            branch_id="", p_t_latent=p, r_long=0.0, u=0.0, per_head_p_t=[p]
        )


def test_make_tasks_groups_by_file():
    examples = [_ex(1, "a.py") for _ in range(6)] + [_ex(0, "b.py") for _ in range(6)]
    tasks = make_tasks(examples, k=3, seed=1, group_by="file")
    assert len(tasks) == 4  # two files x two groups of three
    for t in tasks:
        assert len(t.candidates) == 3
        assert len({c.file for c in t.candidates}) == 1  # homogeneous file


def test_make_tasks_drops_singletons():
    # 4 in one file at k=3 -> one group of 3, the leftover 1 is dropped.
    tasks = make_tasks([_ex(1) for _ in range(4)], k=3, seed=0)
    assert len(tasks) == 1
    assert len(tasks[0].candidates) == 3


def test_oracle_selects_passing_and_abstains():
    tasks = [
        Task([_ex(1), _ex(0), _ex(0)]),  # mixed -> execute the passing one
        Task([_ex(1), _ex(1)]),          # all pass -> execute a passing one
        Task([_ex(0), _ex(0)]),          # all fail -> abstain
    ]
    rep = evaluate_arbiter(tasks, _OracleEnsemble())
    assert rep.n_tasks == 3
    assert rep.execute_precision == 1.0   # never executes a failing branch
    assert rep.selection_recall == 1.0    # always finds the passing branch
    assert rep.safe_abstain == 1.0        # abstains when nothing passes


def test_oracle_never_false_executes_on_all_fail_tasks():
    tasks = [Task([_ex(0), _ex(0), _ex(0)]) for _ in range(5)]
    rep = evaluate_arbiter(tasks, _OracleEnsemble())
    assert rep.n_executed == 0
    assert rep.safe_abstain == 1.0
    assert rep.selection_recall is None  # no task had a passing branch
    assert rep.execute_precision is None  # nothing executed


def test_real_heuristic_metrics_are_valid_ranges():
    examples = [
        _ex(i % 2, file="a.py", complexity=0.1 * (i % 5)) for i in range(12)
    ]
    tasks = make_tasks(examples, k=3, seed=2)
    rep = evaluate_arbiter(tasks, LatentEnsemble())
    assert rep.n_tasks > 0
    assert 0.0 <= rep.execute_rate <= 1.0
    for m in (rep.execute_precision, rep.selection_recall, rep.safe_abstain):
        assert m is None or 0.0 <= m <= 1.0
