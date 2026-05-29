"""Tests for LLM->GNN distillation (Phase 10.3). Offline: a fake simulator and
synthetic rows, so no network/GPU/sandbox is touched."""

import pytest

from nse.config import ROOT
from nse.data.dataset import LabeledExample
from nse.data.distill import attach_soft_labels, distill_agreement
from nse.data.mutate import generate_mutations
from nse.models.latent_model import _TORCH, LatentEnsemble
from nse.orchestrator.schemas import SimulatorPrediction

SEED = ROOT / "nse" / "sandbox_repos" / "mutation_seed"


class FakeSim:
    """Returns a fixed teacher probability for every branch."""

    def __init__(self, p: float = 0.7) -> None:
        self.p = p

    def simulate(self, task, context, branches):
        return {
            b.branch_id: SimulatorPrediction(
                branch_id=b.branch_id, p_t_sim=self.p, explanation="x"
            )
            for b in branches
        }


def test_attach_soft_labels_matches_by_file_and_description():
    rel = "mathx.py"
    muts = generate_mutations((SEED / rel).read_text(encoding="utf-8"))[:5]
    rows = [
        LabeledExample([0.0] * 6, 0, True, m.kind, m.description, rel, "x", 0.0)
        for m in muts
    ]
    attached = attach_soft_labels(rows, SEED, [rel], FakeSim(0.7))
    assert attached >= len(rows)
    assert all(r.soft_label == pytest.approx(0.7) for r in rows)


def test_attach_leaves_unmatched_rows_untouched():
    rel = "mathx.py"
    row = LabeledExample([0.0] * 6, 1, False, "k", "no-such-mutation", rel, "x", 0.0)
    attach_soft_labels([row], SEED, [rel], FakeSim(0.7))
    assert row.soft_label is None  # description never produced by the mutator


def test_distill_agreement_none_without_soft_labels():
    rows = [LabeledExample([0.0] * 6, 1, False, "k", "d", "m.py", "x", 0.0)]
    assert distill_agreement(LatentEnsemble(), rows) is None


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_distillation_pulls_p_t_toward_teacher(monkeypatch):
    """Rows labeled 0 but with a teacher soft label of 0.9: with the distillation
    term active, the GNN's p_t is pulled up toward the teacher versus an
    otherwise-identical model trained without a soft label. The weight is raised
    so the teacher's pull is unambiguous against the hard-label BCE."""
    import nse.models.train as T

    monkeypatch.setattr(T, "DISTILL_WEIGHT", 5.0)
    feats = [1.0, 1.0, 1.0, 6.0, 0.1, 0.0]
    plain = [
        LabeledExample(feats, 0, True, "k", "d", "m.py", "x", 0.0) for _ in range(16)
    ]
    distilled = [
        LabeledExample(feats, 0, True, "k", "d", "m.py", "x", 0.0, soft_label=0.9)
        for _ in range(16)
    ]

    def mean_p(model, rows):
        ens = LatentEnsemble(model=model)
        return sum(ens.predict_from_features(r.features).p_t_latent for r in rows) / len(rows)

    p_plain = mean_p(T.train(plain, epochs=150, seed=0), plain)
    p_distill = mean_p(T.train(distilled, epochs=150, seed=0), distilled)

    assert p_distill > p_plain + 0.2  # teacher (0.9) pulls p_t up despite the 0 label
    assert 0.0 <= p_distill <= 1.0
