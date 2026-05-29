"""LLM->GNN distillation (Phase 10.3).

The cheap GNN's durable role is to *track the expensive LLM*. We transfer the
simulator's judgment into the GNN by attaching its ``p_t_sim`` as a **soft label**
on training rows; :func:`nse.models.train.train` then adds a distillation MSE that
pulls the GNN's ``p_t`` toward the teacher (gated to rows that carry one, so
undistilled training is unchanged).

:func:`attach_soft_labels` regenerates the seed mutants *deterministically* (the
same order the dataset builder used), has the simulator judge them, and matches
the judgments back to dataset rows by ``(file, mutation-description)`` — so it
enriches an existing mutation dataset **without re-running the sandbox**. (The
seed mutants are regenerable; real-corpus distillation would instead need the
patch text stored per row.)

The simulator is passed in (duck-typed ``.simulate(task, context, branches) ->
{branch_id: SimulatorPrediction}``) so this is testable offline with a fake.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from nse.agents.base_client import AgentError
from nse.data.dataset import LabeledExample, _complexity, _unified_diff
from nse.data.mutate import generate_mutations, normalized_source
from nse.memory_graph.context import get_prompt_context
from nse.memory_graph.graph_db import MemoryGraph
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator.schemas import PlannerBranch

DISTILL_TASK = "Will the patched code still pass its existing test suite?"


def attach_soft_labels(
    examples: list[LabeledExample],
    seed_repo: Path | str,
    sources: list[str],
    simulator,
    task: str = DISTILL_TASK,
    batch_size: int = 5,
) -> int:
    """Set ``soft_label`` on ``examples`` from the simulator's ``p_t_sim``.

    Returns the number of rows enriched. Regenerates each source's mutants, has
    the simulator judge them in batches, and matches by ``(file, description)``;
    unmatched rows keep ``soft_label = None``.
    """
    seed_repo = Path(seed_repo)
    index: dict[tuple[str, str], LabeledExample] = {
        (e.file, e.description): e for e in examples
    }
    mg = MemoryGraph(seed_repo).build_graph()
    attached = 0
    for rel in sources:
        original = (seed_repo / rel).read_text(encoding="utf-8")
        baseline = normalized_source(original)
        context = get_prompt_context(mg, [rel], depth=1)
        branches: list[PlannerBranch] = []
        bmap: dict[str, str] = {}  # branch_id -> mutation description
        for mut in generate_mutations(original):
            diff = _unified_diff(baseline, mut.mutated_src, rel)
            bid = str(uuid.uuid4())
            branches.append(
                PlannerBranch(
                    branch_id=bid,
                    strategy=mut.description,
                    edited_files=[rel],
                    patch_preview=diff,
                    expected_complexity=_complexity(diff),
                    planner_confidence=0.5,
                )
            )
            bmap[bid] = mut.description
        for i in range(0, len(branches), batch_size):
            try:
                preds = simulator.simulate(task, context, branches[i : i + batch_size])
            except AgentError:
                continue  # a flaky batch leaves those rows undistilled, not a crash
            for bid, pred in preds.items():
                ex = index.get((rel, bmap.get(bid, "")))
                if ex is not None:
                    ex.soft_label = float(pred.p_t_sim)
                    attached += 1
    return attached


def distill_agreement(
    ensemble: LatentEnsemble, examples: list[LabeledExample]
) -> float | None:
    """Mean ``|GNN p_t - teacher soft label|`` over soft-labeled rows (lower = the
    GNN tracks the teacher more closely). ``None`` when no soft labels are present."""
    rows = [e for e in examples if e.soft_label is not None]
    if not rows:
        return None
    err = 0.0
    for e in rows:
        p = ensemble.predict_from_features(
            e.features, node_features=e.node_features, edge_index=e.edge_index
        ).p_t_latent
        err += abs(p - float(e.soft_label))
    return err / len(rows)
