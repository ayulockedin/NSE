"""Labeled dataset construction.

For every mutant of every seed source file we:
  1. build a real unified diff and wrap it in a ``PlannerBranch``,
  2. extract the 6-dim patch-feature vector the latent model consumes,
  3. write the mutant into a throwaway copy of the repo,
  4. run that copy through the sandbox and record the *true* ``tests_passed``.

The label is ground truth from the verified sandbox, not the mutation's
``assumed_breaking`` hint. Examples serialise to JSONL for offline training.
"""

from __future__ import annotations

import difflib
import json
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nse.data.mutate import generate_mutations, normalized_source
from nse.memory_graph.graph_db import MemoryGraph
from nse.models.cpg_features import blast_radius, build_cpg_features
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator.executor import run_sandbox
from nse.orchestrator.schemas import PlannerBranch


@dataclass
class LabeledExample:
    """One (features -> label) training row with provenance."""

    features: list[float]   # the 6-dim patch-feature vector
    label: int              # tests_passed from the sandbox (ground truth, 0/1)
    assumed_breaking: bool  # the mutation's hint, for diagnostics only
    kind: str               # mutation kind
    description: str
    file: str
    sandbox_mode: str       # "docker" | "local_unsafe"
    runtime: float
    # CPG-lite graph centered on the edited function (depth-1). Defaults keep
    # older datasets / synthetic rows loadable.
    node_features: list[list[float]] = field(default_factory=list)
    edge_index: list[list[int]] = field(default_factory=lambda: [[], []])
    # Structural long-term-risk target for the r_long head: blast radius of the
    # edited function (normalized transitive caller count), in [0, 1].
    r_long_target: float = 0.0
    # Training sample weight (Phase 8.1). Seed mutants are 1.0; replayed
    # deployment examples are up-weighted by how wrong the model was on them.
    weight: float = 1.0
    # Teacher (LLM simulator) probability for distillation (Phase 10.3). None =
    # no teacher signal for this row -> the distillation loss term is skipped.
    soft_label: float | None = None


def _complexity(diff: str) -> float:
    """Map a diff's changed-line count into [0,1] for ``expected_complexity``."""
    changed = sum(
        1
        for ln in diff.splitlines()
        if (ln.startswith("+") and not ln.startswith("+++"))
        or (ln.startswith("-") and not ln.startswith("---"))
    )
    return min(1.0, changed / 20.0)


def _unified_diff(original: str, mutated: str, rel: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            mutated.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )


def build_examples(
    repo_dir: Path | str,
    source_files: list[str],
    force_local: bool = False,
    full: bool = True,
) -> list[LabeledExample]:
    """Generate and label every mutant of ``source_files`` under ``repo_dir``.

    ``force_local=True`` skips Docker (useful on Windows / CI); for *trusted*
    labels run with Docker available and ``force_local=False``.
    """
    repo_dir = Path(repo_dir)
    examples: list[LabeledExample] = []

    # One CPG-lite graph over the pristine repo; the depth-1 subgraph and blast
    # radius are per edited *function*, so cache by (file, function).
    mg = MemoryGraph(repo_dir).build_graph()
    cpg_cache: dict[tuple[str, str], tuple[list[list[float]], list[list[int]]]] = {}
    risk_cache: dict[tuple[str, str], float] = {}

    for rel in source_files:
        original = (repo_dir / rel).read_text(encoding="utf-8")
        # Diff against the normalized baseline so the feature vector reflects
        # only the mutation, not ast.unparse's whole-file reformatting.
        baseline = normalized_source(original)
        for mut in generate_mutations(original):
            key = (rel, mut.function)
            node_features, edge_index = cpg_cache.setdefault(
                key,
                build_cpg_features(
                    mg, [rel], center_functions=[mut.function] if mut.function else None
                ),
            )
            r_long_target = risk_cache.setdefault(
                key,
                blast_radius(mg, f"{rel}::{mut.function}") if mut.function else 0.0,
            )
            diff = _unified_diff(baseline, mut.mutated_src, rel)
            branch = PlannerBranch(
                branch_id=str(uuid.uuid4()),
                strategy=mut.description,
                edited_files=[rel],
                patch_preview=diff,
                expected_complexity=_complexity(diff),
                planner_confidence=0.5,
            )
            feats = LatentEnsemble.patch_features(branch)

            with tempfile.TemporaryDirectory(prefix="nse_mut_") as td:
                snapshot = Path(td) / "repo"
                shutil.copytree(repo_dir, snapshot)
                (snapshot / rel).write_text(mut.mutated_src, encoding="utf-8")
                run = run_sandbox(
                    snapshot,
                    edited_files=[rel],
                    full=full,
                    force_local=force_local,
                )

            examples.append(
                LabeledExample(
                    features=feats,
                    label=int(run.tests_passed),
                    assumed_breaking=mut.assumed_breaking,
                    kind=mut.kind,
                    description=mut.description,
                    file=rel,
                    sandbox_mode=run.mode,
                    runtime=run.runtime,
                    node_features=node_features,
                    edge_index=edge_index,
                    r_long_target=r_long_target,
                )
            )
    return examples


def write_jsonl(examples: list[LabeledExample], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(asdict(ex)) + "\n")
    return path


def read_jsonl(path: Path | str) -> list[LabeledExample]:
    path = Path(path)
    out: list[LabeledExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(LabeledExample(**json.loads(line)))
    return out
