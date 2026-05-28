"""Train the latent transition model on mutation-labeled data.

Pipeline:
  build/load JSONL examples -> stratified train/held-out split -> fit the
  M-head ensemble (BCE on p_t, per-head bootstrap for diversity) -> report
  held-out metrics for the heuristic baseline vs the trained model -> save
  weights to nse/models/weights/latent.pt.

The held-out split is the honesty mechanism: a trained model only "wins" if it
beats the heuristic on data it never saw.

Run:
    python -m nse.models.train --force-local            # build + train
    python -m nse.models.train --rebuild --epochs 300   # regenerate dataset
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from nse.config import ROOT
from nse.data.dataset import LabeledExample, build_examples, read_jsonl, write_jsonl
from nse.eval.harness import evaluate_ensemble
from nse.models.latent_model import (
    DEFAULT_WEIGHTS_PATH,
    LatentEnsemble,
    save_model,
    _TORCH,
)

SEED_REPO = ROOT / "nse" / "sandbox_repos" / "mutation_seed"
SEED_SOURCES = ["mathx.py", "strops.py", "listops.py"]
DATASET_PATH = ROOT / "nse" / "data" / "datasets" / "mutation_seed.jsonl"


# ──────────────────────────── data plumbing ────────────────────────────


def load_or_build(force_local: bool, rebuild: bool) -> list[LabeledExample]:
    if DATASET_PATH.exists() and not rebuild:
        return read_jsonl(DATASET_PATH)
    examples = build_examples(SEED_REPO, SEED_SOURCES, force_local=force_local)
    write_jsonl(examples, DATASET_PATH)
    return examples


def stratified_split(
    examples: list[LabeledExample], test_frac: float, seed: int
) -> tuple[list[LabeledExample], list[LabeledExample]]:
    """Split keeping the pass/fail ratio in both halves."""
    rng = random.Random(seed)
    pos = [e for e in examples if e.label == 1]
    neg = [e for e in examples if e.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def cut(rows: list[LabeledExample]) -> tuple[list, list]:
        n_test = max(1, int(round(len(rows) * test_frac))) if rows else 0
        return rows[n_test:], rows[:n_test]

    tr_p, te_p = cut(pos)
    tr_n, te_n = cut(neg)
    train = tr_p + tr_n
    test = te_p + te_n
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# ──────────────────────────────── training ─────────────────────────────


def train(
    examples: list[LabeledExample],
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
):
    """Fit the ensemble on ``examples``. Returns the trained ``LatentModel``."""
    import torch
    import torch.nn.functional as F

    from nse.models.latent_model import LatentModel

    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    feats = torch.tensor([e.features for e in examples], dtype=torch.float32)
    labels = torch.tensor([e.label for e in examples], dtype=torch.float32)
    n = feats.size(0)

    # Single-node graph per example; the GNN slots in real CPG features later.
    g_x = torch.zeros((n, 32))
    g_x[:, 0] = feats[:, 4]
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.arange(n, dtype=torch.long)

    model = LatentModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Fixed per-head bootstrap masks so the M heads see different subsets and
    # disagree off-distribution -> ensemble variance becomes a real uncertainty.
    head_masks = [
        (torch.rand(n, generator=rng) < 0.8).float() for _ in range(model_M(model))
    ]

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        outs = model(g_x, edge_index, batch, feats)  # list[M] of (n, 2), sigmoid'd
        loss = torch.zeros(())
        for head_out, mask in zip(outs, head_masks):
            p_t = head_out[:, 0].clamp(1e-6, 1 - 1e-6)
            per = F.binary_cross_entropy(p_t, labels, reduction="none")
            loss = loss + (per * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / len(outs)
        loss.backward()
        opt.step()

    model.eval()
    return model


def model_M(model) -> int:
    return len(model.heads)


# ────────────────────────────────── CLI ────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="regenerate dataset")
    parser.add_argument("--force-local", action="store_true", help="label without Docker")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args(argv)

    if not _TORCH:
        print("torch not installed — cannot train.")
        return 1

    print("Loading/building dataset ...")
    examples = load_or_build(args.force_local, args.rebuild)
    passed = sum(e.label for e in examples)
    print(f"  {len(examples)} examples | {passed} pass / {len(examples) - passed} fail")

    train_set, test_set = stratified_split(examples, args.test_frac, args.seed)
    print(f"  split: {len(train_set)} train / {len(test_set)} held-out")
    if not test_set or len({e.label for e in test_set}) < 2:
        print("  WARNING: held-out set lacks both classes; metrics may be degenerate.")

    print(f"Training ensemble for {args.epochs} epochs ...")
    model = train(train_set, epochs=args.epochs, lr=args.lr, seed=args.seed)

    base = evaluate_ensemble(test_set, LatentEnsemble())          # heuristic
    trained = evaluate_ensemble(test_set, LatentEnsemble(model=model))

    print("\n=== HELD-OUT COMPARISON ===")
    print("[heuristic baseline]")
    print(base.render())
    print("\n[trained GNN ensemble]")
    print(trained.render())

    auc_b = base.auc if base.auc is not None else float("nan")
    auc_t = trained.auc if trained.auc is not None else float("nan")
    verdict = "WIN" if (trained.auc or 0) > (base.auc or 0) else "no improvement"
    print(
        f"\nAUC: {auc_b:.3f} -> {auc_t:.3f} | Brier: {base.brier:.3f} -> "
        f"{trained.brier:.3f}  [{verdict}]"
    )

    out = save_model(model, args.out)
    print(f"weights -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
