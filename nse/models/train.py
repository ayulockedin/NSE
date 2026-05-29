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
SEED_SOURCES = [
    "mathx.py",
    "strops.py",
    "listops.py",
    "geometry.py",
    "stats.py",
    "banking.py",
]
DATASET_PATH = ROOT / "nse" / "data" / "datasets" / "mutation_seed.jsonl"

# Weight of the r_long (blast-radius) regression relative to the p_t BCE term.
R_LONG_WEIGHT = 0.5


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


def _batch_graphs(examples: list[LabeledExample]):
    """Pack each example's CPG-lite graph into one disjoint batched graph.

    Returns ``(g_x, edge_index, batch)`` where node indices in ``edge_index``
    are offset per example and ``batch[k]`` names the example that node ``k``
    belongs to. Examples without a stored graph get a single neutral node.
    """
    import torch

    from nse.models.cpg_features import CPG_NODE_DIM

    all_x: list[list[float]] = []
    src: list[int] = []
    dst: list[int] = []
    batch: list[int] = []
    offset = 0
    for i, ex in enumerate(examples):
        nf = ex.node_features or [[0.0] * CPG_NODE_DIM]
        ei = ex.edge_index or [[], []]
        all_x.extend(nf)
        for a, b in zip(ei[0], ei[1]):
            src.append(a + offset)
            dst.append(b + offset)
        batch.extend([i] * len(nf))
        offset += len(nf)

    g_x = torch.tensor(all_x, dtype=torch.float32)
    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.empty((2, 0), dtype=torch.long)
    )
    return g_x, edge_index, torch.tensor(batch, dtype=torch.long)


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
    r_targets = torch.tensor(
        [e.r_long_target for e in examples], dtype=torch.float32
    )
    n = feats.size(0)

    # Batch the per-example CPG-lite graphs into one disjoint graph so the GNN's
    # mean-readout produces exactly one embedding per example (aligned with the
    # patch features and labels by index).
    g_x, edge_index, batch = _batch_graphs(examples)

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
            # Supervise the r_long head on the structural blast-radius target so
            # it carries real signal (the arbiter penalizes high r_long) instead
            # of drifting untrained.
            r_loss = (head_out[:, 1] - r_targets) ** 2
            combined = per + R_LONG_WEIGHT * r_loss
            loss = loss + (combined * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / len(outs)
        loss.backward()
        opt.step()

    model.eval()
    return model


def model_M(model) -> int:
    return len(model.heads)


# ──────────────────────────── cross-validation ─────────────────────────


def stratified_kfold(
    examples: list[LabeledExample], k: int, seed: int
) -> list[list[int]]:
    """Partition example *indices* into k folds, preserving class balance."""
    rng = random.Random(seed)
    pos = [i for i, e in enumerate(examples) if e.label == 1]
    neg = [i for i, e in enumerate(examples) if e.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds: list[list[int]] = [[] for _ in range(k)]
    for grp in (pos, neg):
        for j, idx in enumerate(grp):
            folds[j % k].append(idx)
    return folds


def grouped_folds_by_file(examples: list[LabeledExample]) -> list[list[int]]:
    """Leave-one-file-out folds. Because the CPG graph is shared by all mutants
    of a file, this is the honest test of *cross-file* generalization: the model
    never sees a test file's graph during training, so it can't memorize it."""
    by_file: dict[str, list[int]] = {}
    for i, e in enumerate(examples):
        by_file.setdefault(e.file, []).append(i)
    return list(by_file.values())


def cross_validate(
    examples: list[LabeledExample],
    k: int = 5,
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
    grouped: bool = False,
):
    """Out-of-fold CV. Trains a fresh model per fold, predicts the held-out fold
    with both the heuristic and the trained ensemble, then scores the *pooled*
    OOF predictions once (stable on a small corpus). Returns ``(base, trained)``
    :class:`EvalReport`s."""
    from nse.eval.harness import report_from_probs

    folds = (
        grouped_folds_by_file(examples)
        if grouped
        else stratified_kfold(examples, k, seed)
    )
    base_oof: list[float] = []
    trained_oof: list[float] = []
    labels_oof: list[int] = []

    for f, test_idx in enumerate(folds):
        test_set = [examples[i] for i in test_idx]
        train_set = [examples[i] for i in range(len(examples)) if i not in set(test_idx)]
        if not test_set or not train_set:
            continue
        model = train(train_set, epochs=epochs, lr=lr, seed=seed)
        base = evaluate_ensemble(test_set, LatentEnsemble())
        trained = evaluate_ensemble(test_set, LatentEnsemble(model=model))
        print(
            f"  fold {f + 1}/{len(folds)} (n={len(test_set):3}): "
            f"AUC {fmt(base.auc)}->{fmt(trained.auc)}  "
            f"Brier {base.brier:.3f}->{trained.brier:.3f}"
        )
        base_oof.extend(predict_probs_for(test_set, LatentEnsemble()))
        trained_oof.extend(predict_probs_for(test_set, LatentEnsemble(model=model)))
        labels_oof.extend(e.label for e in test_set)

    return report_from_probs(base_oof, labels_oof), report_from_probs(
        trained_oof, labels_oof
    )


def predict_probs_for(examples, ensemble) -> list[float]:
    from nse.eval.harness import predict_probs

    return predict_probs(examples, ensemble)


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _r_long_mse(ensemble: LatentEnsemble, examples: list[LabeledExample]) -> float:
    """Mean squared error of the ensemble's r_long vs the blast-radius target."""
    if not examples:
        return float("nan")
    se = 0.0
    for e in examples:
        pred = ensemble.predict_from_features(
            e.features, node_features=e.node_features, edge_index=e.edge_index
        )
        se += (pred.r_long - e.r_long_target) ** 2
    return se / len(examples)


# ────────────────────────────────── CLI ────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="regenerate dataset")
    parser.add_argument("--force-local", action="store_true", help="label without Docker")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cv", type=int, default=0, help="k for stratified k-fold CV (0=off)")
    parser.add_argument(
        "--cv-grouped",
        action="store_true",
        help="also run leave-one-file-out CV (honest cross-file generalization)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args(argv)

    if not _TORCH:
        print("torch not installed — cannot train.")
        return 1

    print("Loading/building dataset ...")
    examples = load_or_build(args.force_local, args.rebuild)
    passed = sum(e.label for e in examples)
    print(f"  {len(examples)} examples | {passed} pass / {len(examples) - passed} fail")

    if args.cv:
        print(f"\n=== {args.cv}-FOLD CROSS-VALIDATION (random, stratified) ===")
        base_cv, trained_cv = cross_validate(
            examples, k=args.cv, epochs=args.epochs, lr=args.lr, seed=args.seed
        )
        print("[heuristic, pooled OOF]")
        print(base_cv.render())
        print("\n[trained GNN, pooled OOF]")
        print(trained_cv.render())

    if args.cv_grouped:
        print("\n=== LEAVE-ONE-FILE-OUT CROSS-VALIDATION ===")
        base_g, trained_g = cross_validate(
            examples, epochs=args.epochs, lr=args.lr, seed=args.seed, grouped=True
        )
        print("[heuristic, pooled OOF]")
        print(base_g.render())
        print("\n[trained GNN, pooled OOF]")
        print(trained_g.render())

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

    # r_long sanity: does the supervised head track the blast-radius target on
    # held-out data, and does an untrained ensemble not?
    ens_t = LatentEnsemble(model=model)
    ens_h = LatentEnsemble()
    r_mse_t = _r_long_mse(ens_t, test_set)
    r_mse_h = _r_long_mse(ens_h, test_set)
    print(
        f"\nr_long MSE vs blast-radius target (held-out): "
        f"heuristic {r_mse_h:.4f} -> trained {r_mse_t:.4f}"
    )

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
