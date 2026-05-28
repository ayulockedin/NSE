"""Generate a mutation dataset from the seed repo and score the latent model.

This is the baseline scoreboard: it measures how well the *current* latent
model (the untrained heuristic, unless trained weights are loaded) predicts the
sandbox ground truth. Any trained GNN must beat these numbers on a held-out
split before it ships.

Run:
    python -m nse.scripts.run_eval                 # docker if available
    python -m nse.scripts.run_eval --force-local   # skip docker (Windows/CI)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from nse.config import ROOT
from nse.data.dataset import build_examples, write_jsonl
from nse.eval.harness import evaluate_ensemble
from nse.models.latent_model import LatentEnsemble

SEED = ROOT / "nse" / "sandbox_repos" / "mutation_seed"
SOURCES = ["mathx.py", "strops.py", "listops.py"]
DATASET_OUT = ROOT / "nse" / "data" / "datasets" / "mutation_seed.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-local",
        action="store_true",
        help="skip Docker and label via the local subprocess runner",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="do not write the JSONL dataset"
    )
    args = parser.parse_args(argv)

    print(f"Generating + labeling mutants of {SOURCES} under {SEED} ...")
    examples = build_examples(SEED, SOURCES, force_local=args.force_local)
    if not examples:
        print("No mutants generated — nothing to evaluate.")
        return 1

    modes = Counter(ex.sandbox_mode for ex in examples)
    kinds = Counter(ex.kind for ex in examples)
    passed = sum(ex.label for ex in examples)
    print(
        f"  {len(examples)} mutants | {passed} pass / {len(examples) - passed} fail "
        f"| sandbox modes: {dict(modes)}"
    )
    print(f"  by kind: {dict(kinds)}")
    if "local_unsafe" in modes:
        print("  NOTE: some labels came from the UNSAFE local runner (no isolation).")

    if not args.no_save:
        write_jsonl(examples, DATASET_OUT)
        print(f"  dataset -> {DATASET_OUT}")

    # No trained weights are loaded here, so this scores the heuristic
    # baseline. Once a GNN is trained, load it into LatentEnsemble(model=...).
    ensemble = LatentEnsemble()
    backend = "neural (trained GNN)" if ensemble.model is not None else "heuristic (untrained)"
    report = evaluate_ensemble(examples, ensemble)
    print()
    print(report.render())
    print(f"\nlatent backend: {backend} -- train a GNN to beat the Brier/ECE above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
