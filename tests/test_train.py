"""Tests for the latent-model training loop.

Uses synthetic LabeledExamples (no sandbox) so the suite stays fast. Skips
cleanly if torch is unavailable.
"""

import random

import pytest

from nse.data.dataset import LabeledExample
from nse.models.latent_model import _TORCH, LatentEnsemble, load_ensemble, save_model
from nse.models.train import stratified_split, train

pytestmark = pytest.mark.skipif(not _TORCH, reason="torch not installed")


def _synthetic(n: int = 80, seed: int = 0) -> list[LabeledExample]:
    """Learnable signal that the heuristic provably cannot use.

    The heuristic only reads feats[4] (complexity). We hold the total changed
    lines — and therefore complexity and tokens_changed — CONSTANT across both
    classes, so the only discriminative signal lives in added vs removed:
      * pure insertion  (added=2, removed=0) -> pass
      * replacement     (added=1, removed=1) -> fail
    A little label noise keeps it from being trivially separable.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        label = rng.choice([0, 1])
        added, removed = (2, 0) if label == 1 else (1, 1)
        if rng.random() < 0.1:  # 10% label noise
            label = 1 - label
        total = added + removed  # == 2 for every example -> constant complexity
        feats = [1.0, float(added), float(removed), float(total) * 3,
                 min(1.0, total / 20.0), 0.0]
        out.append(
            LabeledExample(feats, label, removed > 0, "synthetic", "d", "m.py",
                           "local_unsafe", 0.0)
        )
    return out


def test_stratified_split_preserves_both_classes():
    examples = _synthetic(60)
    train_set, test_set = stratified_split(examples, test_frac=0.3, seed=1)
    assert len(train_set) + len(test_set) == len(examples)
    assert {e.label for e in train_set} == {0, 1}
    assert {e.label for e in test_set} == {0, 1}


def test_training_beats_heuristic_on_learnable_signal():
    from nse.eval.harness import evaluate_ensemble

    data = _synthetic(120)
    train_set, test_set = stratified_split(data, test_frac=0.3, seed=2)
    model = train(train_set, epochs=200, seed=2)

    base = evaluate_ensemble(test_set, LatentEnsemble())
    trained = evaluate_ensemble(test_set, LatentEnsemble(model=model))

    # The heuristic ignores added/removed (signal is orthogonal to complexity),
    # so it cannot rank these examples — the trained model must do strictly
    # better on both discrimination (AUC) and calibration (Brier).
    assert trained.auc is not None
    assert trained.auc > base.auc
    assert trained.brier < base.brier


def test_predictions_are_valid_probabilities():
    data = _synthetic(60)
    model = train(data, epochs=50, seed=3)
    ens = LatentEnsemble(model=model)
    for ex in data[:10]:
        pred = ens.predict_from_features(ex.features)
        assert 0.0 <= pred.p_t_latent <= 1.0
        assert pred.u >= 0.0


def test_save_and_load_roundtrip(tmp_path):
    data = _synthetic(40)
    model = train(data, epochs=20, seed=4)
    path = save_model(model, tmp_path / "latent.pt")
    assert path.exists()

    ens = load_ensemble(path)
    assert ens.model is not None
    # Loaded model reproduces the in-memory model's predictions.
    in_mem = LatentEnsemble(model=model)
    feats = data[0].features
    assert abs(
        ens.predict_from_features(feats).p_t_latent
        - in_mem.predict_from_features(feats).p_t_latent
    ) < 1e-5
