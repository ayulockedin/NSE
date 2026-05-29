"""Real-defect resolution eval (Phase 11.4) — report **% resolved**.

The mutation-AUC scoreboard measures discrimination on synthetic edits. The
field's real metric is different and blunter: *given a real bug, does the system
choose to apply the real fix?* This module answers that on the verified pairs
mined by :mod:`nse.data.git_mining`.

Each mined defect contributes one ``real_fix`` example (label 1, the patch that
turns the suite green) and one ``real_regression`` example (label 0). We regroup
them into per-defect **tasks**, score every candidate with the latent ensemble,
route each through the *canonical* arbiter (the same ``aggregate_p_t`` ->
``decide`` -> ``select_best`` path the orchestrator uses — no parallel decision
logic), and ask whether the branch the arbiter would EXECUTE is the real fix.

Metrics
-------
* **resolved_rate**  — defects where the arbiter executes the real fix
  (the toy->real headline: a model that under-scores real patches resolves few).
* **false_fix_rate** — among defects it acted on, the share where it executed a
  regression instead of the fix (the safety failure).
* **abstain_rate**   — defects it declined to act on at all (a miss here, since a
  real fix always exists in the task).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from nse.data.dataset import LabeledExample
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator import arbiter
from nse.orchestrator.schemas import BranchPrediction, Routing


def make_defect_tasks(examples: list[LabeledExample]) -> list[list[LabeledExample]]:
    """Group mined examples into per-defect candidate lists.

    Pairing key is the short SHA that :func:`nse.data.git_mining.build_real_examples`
    leads each description with. A task is kept only if it contains the real fix
    (label 1) — without it there is nothing to resolve.
    """
    groups: "OrderedDict[str, list[LabeledExample]]" = OrderedDict()
    for ex in examples:
        if not ex.kind.startswith("real_"):
            continue
        key = ex.description.split(" ", 1)[0]
        groups.setdefault(key, []).append(ex)
    return [g for g in groups.values() if any(c.label == 1 for c in g)]


@dataclass
class ResolutionReport:
    n_defects: int
    n_resolved: int
    n_acted: int
    n_false_fix: int
    resolved_rate: float
    false_fix_rate: float | None   # among acted
    abstain_rate: float

    def render(self) -> str:
        ff = "n/a" if self.false_fix_rate is None else f"{self.false_fix_rate:.3f}"
        return (
            "-- Real-defect resolution --------------------------\n"
            f"  defects            : {self.n_defects}\n"
            f"  % resolved         : {self.resolved_rate:.3f}  "
            f"({self.n_resolved} executed the real fix)\n"
            f"  false-fix rate     : {ff}  "
            f"({self.n_false_fix} executed a regression | acted)\n"
            f"  abstain rate       : {self.abstain_rate:.3f}  "
            f"({self.n_defects - self.n_acted} declined to act)\n"
            "----------------------------------------------------"
        )


def evaluate_resolution(
    tasks: list[list[LabeledExample]],
    ensemble: LatentEnsemble | None = None,
    sim_p_t: float = 0.5,
    r_critic: float = 0.0,
    c_planner: float = 0.8,
    recalibrator=None,
) -> ResolutionReport:
    """Score per-defect tasks and report resolution. Simulator/critic are held
    neutral so the latent model + arbiter drive the decision (isolating the
    toy->real effect on the GNN)."""
    ensemble = ensemble or LatentEnsemble()
    n_resolved = n_acted = n_false_fix = 0

    for candidates in tasks:
        preds: list[BranchPrediction] = []
        label_of: dict[int, int] = {}
        for i, c in enumerate(candidates):
            lat = ensemble.predict_from_features(
                c.features, node_features=c.node_features, edge_index=c.edge_index
            )
            pred = BranchPrediction(
                branch_id=str(i),
                p_c=1,  # mined patches are valid Python; symbolic gate assumed passed
                p_t=arbiter.aggregate_p_t(lat.p_t_latent, sim_p_t),
                u=lat.u,
                r_critic=r_critic,
                r_long=lat.r_long,
                c_planner=c_planner,
            )
            arbiter.decide(pred, recalibrator=recalibrator)
            preds.append(pred)
            label_of[id(pred)] = c.label

        best = arbiter.select_best(preds)
        if best is None or best.routing != Routing.EXECUTE:
            continue
        n_acted += 1
        if label_of[id(best)] == 1:
            n_resolved += 1
        else:
            n_false_fix += 1

    n = len(tasks)
    return ResolutionReport(
        n_defects=n,
        n_resolved=n_resolved,
        n_acted=n_acted,
        n_false_fix=n_false_fix,
        resolved_rate=(n_resolved / n) if n else 0.0,
        false_fix_rate=(n_false_fix / n_acted) if n_acted else None,
        abstain_rate=((n - n_acted) / n) if n else 0.0,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from nse.data.dataset import read_jsonl
    from nse.models.latent_model import load_ensemble

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, type=Path, help="mined real-defect JSONL")
    parser.add_argument(
        "--weights", type=Path, default=None, help="trained latent weights (default: heuristic)"
    )
    args = parser.parse_args(argv)

    examples = read_jsonl(args.real)
    tasks = make_defect_tasks(examples)
    print(f"{len(tasks)} defect tasks from {len(examples)} mined examples\n")

    print("[heuristic ensemble]")
    print(evaluate_resolution(tasks, LatentEnsemble()).render())
    if args.weights:
        print("\n[trained ensemble]")
        print(evaluate_resolution(tasks, load_ensemble(args.weights, strict=False)).render())
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
