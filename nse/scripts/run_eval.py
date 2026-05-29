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
from nse.eval.arbiter_eval import evaluate_arbiter, make_tasks
from nse.eval.harness import evaluate_ensemble
from nse.models.latent_model import LatentEnsemble, load_ensemble

SEED = ROOT / "nse" / "sandbox_repos" / "mutation_seed"
SOURCES = ["mathx.py", "strops.py", "listops.py", "geometry.py", "stats.py", "banking.py"]
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

    # Compare the untrained heuristic against the trained GNN (if weights exist).
    heuristic = LatentEnsemble()
    trained = load_ensemble(strict=False)
    has_trained = trained.model is not None

    print("\n[latent model — heuristic baseline]")
    print(evaluate_ensemble(examples, heuristic).render())
    if has_trained:
        print("\n[latent model — trained GNN]")
        print(evaluate_ensemble(examples, trained).render())

    # Arbiter-level selection eval: can it pick a passing branch from a group,
    # and abstain when none passes?
    tasks = make_tasks(examples)
    print(f"\n=== ARBITER SELECTION ({len(tasks)} tasks) ===")
    print("\n[arbiter — heuristic latent]")
    print(evaluate_arbiter(tasks, heuristic).render())
    if has_trained:
        print("\n[arbiter — trained latent]")
        print(evaluate_arbiter(tasks, trained).render())
    else:
        print("\n(no trained weights found — run `python -m nse.models.train` first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
