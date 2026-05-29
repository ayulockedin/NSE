"""Cost model + cost-aware acquisition (Phase 10.2).

NSE's thesis is that it allocates scarce *verification* budget under uncertainty.
To do that it has to price the tiers it can spend on:

* **GNN screen**  — microseconds; effectively free.
* **LLM judge**   — priced per token (the Simulator/Critic call).
* **sandbox run** — the expensive tier (process isolation + a test suite).

Costs are abstract additive "cost units". The defaults make one sandbox run the
unit (``1.0``) and an LLM judge a configurable fraction, so the cascade can reason
about savings without committing to dollars. Swap in real $/token + $/run later.

**Acquisition (10.2):** when several branches *could* be sandboxed but the budget
allows only some, spend where the **expected information gain per unit cost** is
highest — the branch whose outcome we're least sure of (binary entropy of ``p_t``
for aleatoric uncertainty, plus the epistemic ensemble variance ``u``), divided by
what the run costs. This is the knob that turns "run k sandboxes" into "spend the
next run where it buys the most decision certainty per dollar".

Pure functions + plain dataclasses: no torch, no I/O, exhaustively unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nse.orchestrator.arbiter import normalize_uncertainty


@dataclass(frozen=True)
class CostModel:
    """Prices each tier in abstract cost units (1 sandbox run == 1.0 by default)."""

    gnn_cost: float = 0.0                 # GNN screen ≈ free
    llm_cost_per_1k_tokens: float = 0.05  # the Simulator/Critic call
    sandbox_cost_per_run: float = 1.0     # the unit

    def gnn(self, n: int = 1) -> float:
        return self.gnn_cost * n

    def llm(self, tokens: int) -> float:
        return self.llm_cost_per_1k_tokens * (tokens / 1000.0)

    def sandbox(self, runs: int = 1) -> float:
        return self.sandbox_cost_per_run * runs


@dataclass
class CostLedger:
    """Accumulates spend per tier across one task (the cascade fills this in)."""

    gnn: float = 0.0
    llm: float = 0.0
    sandbox: float = 0.0

    def add_gnn(self, c: float) -> None:
        self.gnn += c

    def add_llm(self, c: float) -> None:
        self.llm += c

    def add_sandbox(self, c: float) -> None:
        self.sandbox += c

    @property
    def total(self) -> float:
        return self.gnn + self.llm + self.sandbox

    def as_dict(self) -> dict[str, float]:
        return {"gnn": self.gnn, "llm": self.llm, "sandbox": self.sandbox, "total": self.total}


def binary_entropy(p: float) -> float:
    """Shannon entropy (bits) of a Bernoulli(``p``). Peaks at ``p = 0.5`` (maximal
    uncertainty about the outcome) and is ``0`` at the certain endpoints."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def info_gain(p_t: float, u: float = 0.0, gamma: float = 1.0) -> float:
    """Expected information from sandboxing a branch — how unsure we are about its
    outcome. Aleatoric term = binary entropy of ``p_t``; epistemic term = the
    normalized ensemble variance ``u`` (model "doesn't know"), weighted by
    ``gamma``. Higher means a sandbox run resolves more uncertainty."""
    return binary_entropy(p_t) + gamma * normalize_uncertainty(u)


def acquisition_score(p_t: float, u: float, cost: float, gamma: float = 1.0) -> float:
    """Information gain per unit cost — the cost-aware acquisition criterion. A free
    run (``cost <= 0``) is infinitely attractive; otherwise it's ``info / cost``."""
    if cost <= 0.0:
        return float("inf")
    return info_gain(p_t, u, gamma) / cost


def rank_by_acquisition(
    items: list[tuple[object, float, float]],
    cost: float,
    gamma: float = 1.0,
) -> list[object]:
    """Order opaque handles by acquisition (desc). ``items`` are
    ``(handle, p_t, u)`` triples; ``cost`` is the per-run sandbox cost. Use this to
    decide *which* branch to spend the next sandbox run on first."""
    scored = [
        (acquisition_score(p_t, u, cost, gamma), i, handle)
        for i, (handle, p_t, u) in enumerate(items)
    ]
    # Sort by score desc; index keeps it stable for equal scores (deterministic).
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [handle for _, _, handle in scored]
