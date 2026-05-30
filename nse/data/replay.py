"""Replay flywheel (Phase 8.1) — turn deployment outcomes into training data.

Every task logs predictions; executed branches log outcomes; the audit loop
re-executes sampled pruned branches. All of those carry a *ground-truth* label.
This module reconstructs them into :class:`LabeledExample`s — the same
(features + per-function CPG graph + r_long target) shape the seed dataset uses —
and **up-weights the ones the model got most wrong**, so retraining sharpens the
model exactly where it failed in the wild (audit-found false negatives especially).

Reconstruction needs the pristine per-task snapshot the orchestrator retained, so
this only harvests tasks whose snapshot still exists.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from nse.data.dataset import LabeledExample
from nse.memory_graph.graph_db import MemoryGraph
from nse.models.cpg_features import blast_radius, build_cpg_features, changed_function_names
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator.patcher import (
    PatchApplyError,
    PatchSafetyError,
    apply_patch_strict_then_fallback,
)
from nse.orchestrator.schemas import PlannerBranch

if TYPE_CHECKING:
    from nse.db.db_client import DBClient


def example_weight(predicted_p_t: float, label: int, scale: float = 4.0) -> float:
    """Higher weight for larger prediction error. A confident-wrong example
    (e.g. predicted 0.1, actually passed) gets ``1 + scale``; a correct,
    confident one stays near 1.0."""
    return 1.0 + scale * abs(predicted_p_t - float(label))


def _snapshot_dir(audit_dir: Path, task_id: str) -> Path:
    return audit_dir / task_id / "repo"


def _reconstruct_example(
    branch: PlannerBranch, snapshot: Path, label: int, predicted_p_t: float
) -> LabeledExample | None:
    """Rebuild a LabeledExample for ``branch`` against its pristine ``snapshot``,
    mirroring the dataset builder + serving featurizer (per-function CPG)."""
    if not branch.edited_files:
        return None
    feats = LatentEnsemble.patch_features(branch)
    mg = MemoryGraph(snapshot).build_graph()

    # Identify the edited function the same way serving does (apply -> diff).
    fns: list[str] = []
    work = Path(tempfile.mkdtemp(prefix="nse_replay_"))
    repo = work / "repo"
    try:
        shutil.copytree(snapshot, repo)
        try:
            apply_patch_strict_then_fallback(
                repo,
                patch_text=branch.patch_preview,
                full_rewrites=branch.full_file_rewrites,
            )
            fns = changed_function_names(snapshot, repo, branch.edited_files)
        except (PatchApplyError, PatchSafetyError):
            fns = []
    finally:
        shutil.rmtree(work, ignore_errors=True)

    node_features, edge_index = build_cpg_features(
        mg, branch.edited_files, center_functions=fns or None
    )
    r_long_target = (
        blast_radius(mg, f"{branch.edited_files[0]}::{fns[0]}") if fns else 0.0
    )
    return LabeledExample(
        features=feats,
        label=int(label),
        assumed_breaking=(int(label) == 0),
        kind="replay",
        description=branch.strategy,
        file=branch.edited_files[0],
        sandbox_mode="replay",
        runtime=0.0,
        node_features=node_features,
        edge_index=edge_index,
        r_long_target=r_long_target,
        weight=example_weight(predicted_p_t, label),
    )


def harvest_training_examples(
    db: "DBClient", audit_dir: Path | str
) -> list[LabeledExample]:
    """Reconstruct weighted training examples from logged deployment outcomes.

    Skips any task whose pristine snapshot is gone (can't rebuild the graph)."""
    audit_dir = Path(audit_dir)
    examples: list[LabeledExample] = []
    for row in db.training_signals():
        snapshot = _snapshot_dir(audit_dir, row["task_id"])
        if not snapshot.exists():
            continue
        try:
            branch = PlannerBranch.model_validate_json(row["planner_json"])
        except ValueError:
            continue
        ex = _reconstruct_example(
            branch, snapshot, int(row["label"]), float(row["p_t"])
        )
        if ex is not None:
            examples.append(ex)
    return examples
