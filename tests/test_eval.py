"""Tests for the dataset builder and the eval harness."""

from pathlib import Path

import pytest

from nse.data.dataset import (
    LabeledExample,
    build_examples,
    read_jsonl,
    write_jsonl,
)
from nse.eval.harness import evaluate_ensemble

# A tiny self-contained repo so labeling stays fast (few sandbox runs).
SRC = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "def gt(a, b):\n"
    "    return a > b\n"
)
TEST = (
    "from m import add, gt\n"
    "\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "\n"
    "def test_gt():\n"
    "    assert gt(3, 1) is True\n"
    "    assert gt(1, 3) is False\n"
)


@pytest.fixture
def seed_repo(tmp_path: Path) -> Path:
    (tmp_path / "m.py").write_text(SRC)
    (tmp_path / "test_m.py").write_text(TEST)
    return tmp_path


def test_build_examples_labels_via_sandbox(seed_repo: Path):
    # force_local keeps this runnable on Windows/CI without Docker.
    examples = build_examples(seed_repo, ["m.py"], force_local=True)
    assert examples
    # Every example carries a 6-dim feature vector and a 0/1 label.
    assert all(len(ex.features) == 6 for ex in examples)
    assert all(ex.label in (0, 1) for ex in examples)
    # The thorough tests must kill at least one breaking mutant ...
    assert any(ex.label == 0 for ex in examples)
    # ... and benign no-ops must survive.
    assert any(ex.label == 1 for ex in examples)


def test_jsonl_roundtrip(tmp_path: Path):
    examples = [
        LabeledExample(
            features=[1.0, 2.0, 0.0, 5.0, 0.1, 0.0],
            label=1,
            assumed_breaking=False,
            kind="benign_noop",
            description="noop in f",
            file="m.py",
            sandbox_mode="local_unsafe",
            runtime=0.12,
        )
    ]
    path = write_jsonl(examples, tmp_path / "d.jsonl")
    back = read_jsonl(path)
    assert back == examples


def test_evaluate_ensemble_perfect_predictions():
    # Heuristic gives high p_t for simple patches; here we hand-craft examples
    # whose features make the heuristic confidently correct, then sanity-check
    # the metric plumbing on perfectly-separable synthetic probs instead.
    examples = [
        LabeledExample([1, 1, 0, 4, 0.0, 0.0], 1, False, "benign", "d", "m.py", "local_unsafe", 0.0),
        LabeledExample([1, 1, 0, 4, 1.0, 0.0], 0, True, "binop", "d", "m.py", "local_unsafe", 0.0),
    ]
    report = evaluate_ensemble(examples)
    assert report.n == 2
    assert report.base_rate == 0.5
    assert 0.0 <= report.brier <= 1.0
    assert 0.0 <= report.ece <= 1.0
    # complexity 0.0 -> high p_t (>=0.5 -> predicts pass);
    # complexity 1.0 -> low p_t (<0.5 -> predicts fail). Heuristic is correct.
    assert report.accuracy == 1.0
    assert report.auc == 1.0


def test_evaluate_empty():
    report = evaluate_ensemble([])
    assert report.n == 0
    assert report.auc is None
