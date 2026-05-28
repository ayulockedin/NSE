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
from dataclasses import asdict, dataclass
from pathlib import Path

from nse.data.mutate import generate_mutations, normalized_source
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

    for rel in source_files:
        original = (repo_dir / rel).read_text(encoding="utf-8")
        # Diff against the normalized baseline so the feature vector reflects
        # only the mutation, not ast.unparse's whole-file reformatting.
        baseline = normalized_source(original)
        for mut in generate_mutations(original):
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
