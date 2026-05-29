"""Arbiter-level (selection) evaluation.

The latent-model harness scores each candidate *in isolation*. But NSE's real
job is to pick ONE branch to execute from a group of competitors — and to
ABSTAIN when none is safe. This module measures that selection quality.

For each synthetic task (a group of candidate patches with known pass/fail
labels) every candidate is scored by the latent ensemble and routed by the
arbiter; ``arbiter.select_best`` then picks the EXECUTE branch (or abstains). We
compare the pick against ground truth.

Simulator/critic signals are held neutral (no offline LLM), so the latent model
+ arbiter math drive the decision — this isolates *selection* skill, which the
per-example AUC cannot see.

Metrics:
  execute_rate      fraction of tasks where the arbiter executed something
  execute_precision among executed tasks, fraction whose pick actually passes
                    (1 - false-execute rate — the safety headline)
  selection_recall  among tasks where a passing branch EXISTS, fraction where the
                    arbiter executed a passing one (did it find the good patch)
  safe_abstain      among tasks with NO passing branch, fraction it abstained on
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from nse.config import SETTINGS
from nse.data.dataset import LabeledExample
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator import arbiter
from nse.orchestrator.schemas import BranchPrediction


@dataclass
class Task:
    """A group of competing candidate patches for one (synthetic) fix."""

    candidates: list[LabeledExample]


def make_tasks(
    examples: list[LabeledExample],
    k: int = SETTINGS.hp.k,
    seed: int = 0,
    group_by: str = "file",
) -> list[Task]:
    """Partition examples into candidate groups of up to ``k``.

    ``group_by="file"`` keeps a task's candidates within one module (realistic:
    competing patches to the same code); ``"none"`` draws k-subsets across the
    whole corpus (more class-mixed groups). Groups smaller than 2 are dropped —
    selection needs at least two options.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[LabeledExample]] = defaultdict(list)
    for e in examples:
        buckets[e.file if group_by == "file" else "all"].append(e)

    tasks: list[Task] = []
    for rows in buckets.values():
        rows = rows[:]
        rng.shuffle(rows)
        for i in range(0, len(rows), k):
            chunk = rows[i : i + k]
            if len(chunk) >= 2:
                tasks.append(Task(chunk))
    return tasks


@dataclass
class ArbiterReport:
    n_tasks: int
    n_executed: int
    n_has_pass: int
    n_no_pass: int
    execute_rate: float
    execute_precision: float | None
    selection_recall: float | None
    safe_abstain: float | None

    @staticmethod
    def _fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    def render(self) -> str:
        return (
            "-- Arbiter selection eval --------------------------\n"
            f"  tasks              : {self.n_tasks}  "
            f"({self.n_has_pass} with a passing branch, {self.n_no_pass} without)\n"
            f"  execute rate       : {self.execute_rate:.3f}  "
            f"({self.n_executed} executed)\n"
            f"  execute precision  : {self._fmt(self.execute_precision)}  "
            "(pick actually passes | executed)\n"
            f"  selection recall   : {self._fmt(self.selection_recall)}  "
            "(found a passing branch | one existed)\n"
            f"  safe abstain       : {self._fmt(self.safe_abstain)}  "
            "(abstained | no passing branch)\n"
            "----------------------------------------------------"
        )


def evaluate_arbiter(
    tasks: list[Task],
    ensemble: LatentEnsemble | None = None,
    sim_p_t: float = 0.5,
    r_critic: float = 0.0,
    c_planner: float = 0.8,
    recalibrator=None,
) -> ArbiterReport:
    ensemble = ensemble or LatentEnsemble()
    n_exec = n_exec_pass = 0
    n_has_pass = n_recall_hit = 0
    n_no_pass = n_safe = 0

    for task in tasks:
        has_pass = any(c.label == 1 for c in task.candidates)
        if has_pass:
            n_has_pass += 1
        else:
            n_no_pass += 1

        preds: list[BranchPrediction] = []
        label_of: dict[int, int] = {}
        for i, c in enumerate(task.candidates):
            lat = ensemble.predict_from_features(
                c.features, node_features=c.node_features, edge_index=c.edge_index
            )
            p_t = arbiter.aggregate_p_t(lat.p_t_latent, sim_p_t)
            pred = BranchPrediction(
                branch_id=str(i),
                p_c=1,  # mutants are valid Python; symbolic gate assumed passed
                p_t=p_t,
                u=lat.u,
                r_critic=r_critic,
                r_long=lat.r_long,
                c_planner=c_planner,
            )
            arbiter.decide(pred, recalibrator=recalibrator)
            preds.append(pred)
            label_of[id(pred)] = c.label

        best = arbiter.select_best(preds)
        if best is None:
            if not has_pass:
                n_safe += 1
            continue

        n_exec += 1
        if label_of[id(best)] == 1:
            n_exec_pass += 1
            if has_pass:
                n_recall_hit += 1

    n = len(tasks)
    return ArbiterReport(
        n_tasks=n,
        n_executed=n_exec,
        n_has_pass=n_has_pass,
        n_no_pass=n_no_pass,
        execute_rate=(n_exec / n) if n else 0.0,
        execute_precision=(n_exec_pass / n_exec) if n_exec else None,
        selection_recall=(n_recall_hit / n_has_pass) if n_has_pass else None,
        safe_abstain=(n_safe / n_no_pass) if n_no_pass else None,
    )
