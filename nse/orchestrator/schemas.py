"""Strict Pydantic schemas for agent I/O and internal records.

These are the authoritative shapes from the build-ready blueprint (section 5).
Every LLM output is validated against one of the agent schemas; on failure the
orchestrator attempts a single auto-repair, then marks the output malformed and
penalises the planner confidence.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────── Agent outputs ────────────────────────────


class PlannerBranch(BaseModel):
    """One candidate implementation strategy produced by the Planner."""

    branch_id: str
    strategy: str
    edited_files: list[str]
    patch_preview: Optional[str] = None          # unified diff, if generated
    full_file_rewrites: Optional[dict[str, str]] = None  # {path: full_text}
    assumptions: list[str] = Field(default_factory=list)
    expected_complexity: float = Field(ge=0.0, le=1.0)
    planner_confidence: float = Field(ge=0.0, le=1.0)


class SimulatorPrediction(BaseModel):
    """Analytic LLM prior on whether tests will pass."""

    branch_id: str
    p_t_sim: float = Field(ge=0.0, le=1.0)
    explanation: str


class CriticReport(BaseModel):
    """Adversarial red-team assessment."""

    branch_id: str
    r_critic: float = Field(ge=0.0, le=1.0)
    identified_failures: list[str] = Field(default_factory=list)
    attack_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("identified_failures", mode="before")
    @classmethod
    def _coerce_failures(cls, v: object) -> list[str]:
        """Real models often return failures as objects or a single string;
        coerce to a list of short strings so a stylistic deviation doesn't fail
        the whole critic call."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if not isinstance(v, list):
            return [str(v)]
        out: list[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(
                    str(item.get("description")
                        or item.get("failure")
                        or item.get("scenario")
                        or item)
                )
            else:
                out.append(str(item))
        return out


# ──────────────────────────── Internal records ────────────────────────────


class PruneReason(str, Enum):
    SYMBOLIC_FAIL = "symbolic_fail"
    LOW_SCORE = "low_score"
    UNSAFE_PATCH = "unsafe_patch"
    MALFORMED = "malformed"
    BUDGET_EXCEEDED = "budget_exceeded"
    ORACLE_CRASH = "oracle_crash"  # property oracle: patch crashes on inputs the original handled


class Routing(str, Enum):
    PRUNE = "prune"
    INCREMENTAL_SANDBOX = "incremental_sandbox"
    HUMAN_REVIEW = "human_review"
    EXECUTE = "execute"


class LatentPrediction(BaseModel):
    """Output of the latent transition model for one branch."""

    branch_id: str
    p_t_latent: float = Field(ge=0.0, le=1.0)
    r_long: float = Field(ge=0.0, le=1.0)
    u: float = Field(ge=0.0)                      # epistemic: variance across heads
    u_aleatoric: float = Field(default=0.0, ge=0.0)  # irreducible: mean head p(1-p)
    per_head_p_t: list[float] = Field(default_factory=list)


class BranchPrediction(BaseModel):
    """All scored signals for a branch — mirrors the ``predictions`` table."""

    branch_id: str
    p_c: int = Field(ge=0, le=1)                  # symbolic compile gate
    p_t_sim: float = Field(default=0.0, ge=0.0, le=1.0)
    p_t_latent: float = Field(default=0.0, ge=0.0, le=1.0)
    p_t: float = Field(default=0.0, ge=0.0, le=1.0)
    u: float = Field(default=0.0, ge=0.0)        # epistemic uncertainty
    u_aleatoric: float = Field(default=0.0, ge=0.0)  # irreducible uncertainty
    r_critic: float = Field(default=0.0, ge=0.0, le=1.0)
    r_long: float = Field(default=0.0, ge=0.0, le=1.0)
    c_planner: float = Field(default=0.0, ge=0.0, le=1.0)
    score: Optional[float] = None
    routing: Optional[Routing] = None
    prune_reason: Optional[PruneReason] = None


class Outcome(BaseModel):
    """Ground-truth sandbox result — mirrors the ``outcomes`` table."""

    branch_id: str
    compiled: int = Field(ge=0, le=1)
    tests_passed: int = Field(ge=0, le=1)
    logs: str = ""
    runtime: float = 0.0


class SymbolicResult(BaseModel):
    """Output of the symbolic filter (Layer 1)."""

    syntax_ok: bool
    type_ok: bool
    deps_ok: bool
    p_c: int = Field(ge=0, le=1)
    details: str = ""

    @field_validator("p_c")
    @classmethod
    def _gate_consistency(cls, v: int) -> int:
        return v
