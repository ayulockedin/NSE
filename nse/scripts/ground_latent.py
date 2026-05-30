"""Ground the shipped latent model on real-defect data (Phase 11 finale).

Curriculum-train (mutation pretrain -> warm-started fine-tune on seed + real) on
ALL available data and save to the deployed weights, so the GNN stops
under-scoring real patches (the toy->real gap). Production trains on everything;
the leakage-free generalization estimate is the separate leave-one-pair-out CV
(see ROADMAP Phase 11), not this script.

Usage:
    python -m nse.scripts.ground_latent --real corpus1.jsonl corpus2.jsonl
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from nse.config import ROOT
from nse.data.dataset import read_jsonl
from nse.models.latent_model import (
    DEFAULT_WEIGHTS_PATH,
    LatentEnsemble,
    _TORCH,
    save_model,
)

SEED_DATASET = ROOT / "nse" / "data" / "datasets" / "mutation_seed.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", nargs="+", required=True, help="mined real-defect JSONL(s)")
    ap.add_argument("--epochs", type=int, default=300, help="pretrain epochs")
    ap.add_argument("--finetune-epochs", type=int, default=200)
    ap.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS_PATH)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)

    if not _TORCH:
        print("torch not installed — cannot train.")
        return 1

    from nse.models.train import curriculum_train  # lazy (imports torch)

    seed = read_jsonl(SEED_DATASET)
    real: list = []
    for p in args.real:
        real.extend(read_jsonl(p))
    real_fix = sum(1 for e in real if e.label == 1)
    print(f"seed mutants: {len(seed)} | real examples: {len(real)} "
          f"({real_fix} fix / {len(real) - real_fix} regression)")
    if not real:
        print("no real examples — refusing to 'ground' on nothing.")
        return 1

    if args.out.exists() and not args.no_backup:
        backup = args.out.with_suffix(".synthetic.pt")
        shutil.copy(args.out, backup)
        print(f"backed up current weights -> {backup}")

    print(f"curriculum: pretrain {args.epochs}e on mutants -> "
          f"fine-tune {args.finetune_epochs}e on seed+real ...")
    model = curriculum_train(
        seed, real, epochs=args.epochs, finetune_epochs=args.finetune_epochs, seed=0
    )
    out = save_model(model, args.out)

    # Sanity: the shipped model still emits valid probabilities.
    ens = LatentEnsemble(model=model)
    p = ens.predict_from_features(seed[0].features).p_t_latent
    assert 0.0 <= p <= 1.0
    print(f"grounded weights -> {out}  (sanity p_t={p:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
