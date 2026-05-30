"""Cost-tiered cascade (Phase 10.1).

The neural core's *durable* role — the one that survives every LLM upgrade — is a
cheap, calibrated **filter** that saves expensive LLM and sandbox calls. This runs
the decision as explicit tiers with early-exit:

    Tier 1  GNN screen   (≈ free)   score every candidate; drop the arbiter's prunes.
    Tier 2  LLM judge    (tokens)   only *survivors* get a Simulator opinion; the
                                     re-aggregated score may prune more.
    Tier 3  sandbox      (a run)    only the single finalist is executed/verified.

The win: the GNN screen means the LLM is never called on branches that were
hopeless anyway, and the sandbox runs once instead of k times. The result records
what each tier cost and how many LLM/sandbox calls the screen *saved* versus the
naive "judge + sandbox every candidate" baseline.

Components are injected as **callables** so this is fully testable offline (the
heuristic ensemble + stubs). It uses the *canonical* arbiter (`aggregate_p_t` →
`decide` → `select_best`) at each tier — there is no parallel decision logic.

`Orchestrator.run_task` implements this exact tiered ordering inline (it needs the
symbolic gate, per-branch coverage, the property oracle, and DB logging woven in),
so this module is the standalone, executable *spec* of the pattern — handy for
offline cascade analysis and as the reference the orchestrator mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, TypeVar

from nse.orchestrator import arbiter
from nse.orchestrator.cost import CostLedger, CostModel
from nse.orchestrator.schemas import BranchPrediction, LatentPrediction, Routing

H = TypeVar("H")  # opaque candidate handle (a PlannerBranch, an id, …)


@dataclass
class TierResult:
    name: str
    n_in: int
    n_out: int
    cost: float


@dataclass
class CascadeResult(Generic[H]):
    finalist: Optional[H]
    verified: Optional[bool]              # sandbox outcome of the finalist (None if none ran)
    survivors: list[H]                    # candidates that reached the finalist pool
    tiers: list[TierResult] = field(default_factory=list)
    ledger: CostLedger = field(default_factory=CostLedger)
    llm_calls_saved: int = 0             # candidates the GNN screen kept off the LLM
    sandbox_runs_saved: int = 0          # candidates we did NOT have to execute

    @property
    def total_cost(self) -> float:
        return self.ledger.total

    def render(self) -> str:
        lines = ["-- Cascade --------------------------------------"]
        for t in self.tiers:
            lines.append(f"  {t.name:14s} {t.n_in:3d} -> {t.n_out:3d}  cost {t.cost:.3f}")
        lines.append(
            f"  saved: {self.llm_calls_saved} LLM call(s), "
            f"{self.sandbox_runs_saved} sandbox run(s)"
        )
        lines.append(
            f"  finalist: {'yes' if self.finalist is not None else 'none'}"
            f"  verified: {self.verified}  total cost {self.total_cost:.3f}"
        )
        lines.append("-------------------------------------------------")
        return "\n".join(lines)


def run_cascade(
    candidates: list[H],
    screen: Callable[[H], LatentPrediction],
    judge: Callable[[H], tuple[float, int]],
    verify: Callable[[H], bool],
    cost_model: Optional[CostModel] = None,
    c_planner: float = 0.8,
    screen_sim: float = 1.0,
    recalibrator: Optional[Callable[[float], float]] = None,
    conformal_threshold: Optional[float] = None,
) -> CascadeResult[H]:
    """Run the three-tier cascade over ``candidates``.

    ``screen(c) -> LatentPrediction``  — the GNN (cheap), called on every candidate.
    ``judge(c)  -> (p_t_sim, tokens)`` — the LLM, called only on screen survivors.
    ``verify(c) -> tests_passed``      — the sandbox, called only on the finalist.

    Tier 1 routes with the *optimistic* simulator prior ``screen_sim`` (default 1.0
    = best-case LLM): since S(B) increases monotonically in p_t, a branch the screen
    prunes could not clear tau under ANY judgment, so it is dropped before any LLM
    spend without hurting recall. Tier 2 re-aggregates with the real simulator
    opinion and may prune further; the arbiter then selects the one finalist.
    """
    cm = cost_model or CostModel()
    ledger = CostLedger()
    n = len(candidates)

    # ── Tier 1: GNN screen (every candidate) ────────────────────────────────
    survivors: list[tuple[H, BranchPrediction]] = []
    for i, c in enumerate(candidates):
        lat = screen(c)
        ledger.add_gnn(cm.gnn())
        pred = BranchPrediction(
            branch_id=str(i),
            p_c=1,  # caller is responsible for the symbolic gate; cascade assumes compiled
            p_t_sim=screen_sim,
            p_t_latent=lat.p_t_latent,
            p_t=arbiter.aggregate_p_t(lat.p_t_latent, screen_sim),
            u=lat.u,
            r_long=lat.r_long,
            c_planner=c_planner,
        )
        arbiter.decide(pred, recalibrator=recalibrator, conformal_threshold=conformal_threshold)
        if pred.routing != Routing.PRUNE:
            survivors.append((c, pred))
    tier1 = TierResult("gnn_screen", n, len(survivors), ledger.gnn)

    # ── Tier 2: LLM judge (survivors only) ──────────────────────────────────
    llm_before = ledger.llm
    judged: list[tuple[H, BranchPrediction]] = []
    for c, pred in survivors:
        p_t_sim, tokens = judge(c)
        ledger.add_llm(cm.llm(tokens))
        pred.p_t_sim = p_t_sim
        pred.p_t = arbiter.aggregate_p_t(pred.p_t_latent, p_t_sim)
        arbiter.decide(pred, recalibrator=recalibrator, conformal_threshold=conformal_threshold)
        if pred.routing != Routing.PRUNE:
            judged.append((c, pred))
    tier2 = TierResult("llm_judge", len(survivors), len(judged), ledger.llm - llm_before)

    # ── Tier 3: sandbox verify (finalist only) ──────────────────────────────
    best = arbiter.select_best([p for _, p in judged])
    finalist: Optional[H] = None
    verified: Optional[bool] = None
    if best is not None:
        finalist = next(c for c, p in judged if p is best)
        verified = verify(finalist)
        ledger.add_sandbox(cm.sandbox())
    tier3 = TierResult("sandbox_verify", len(judged), 1 if finalist is not None else 0, ledger.sandbox)

    return CascadeResult(
        finalist=finalist,
        verified=verified,
        survivors=[c for c, _ in judged],
        tiers=[tier1, tier2, tier3],
        ledger=ledger,
        # The screen kept (n - survivors) candidates off the LLM; the cascade ran
        # the sandbox at most once instead of on every candidate.
        llm_calls_saved=n - len(survivors),
        sandbox_runs_saved=n - (1 if finalist is not None else 0),
    )
