"""Central configuration & immutable hyperparameters.

These constants are the canonical defaults from the build-ready blueprint
(section 2 & 4.5). The scoring math that consumes them lives in
``nse.orchestrator.arbiter`` and must not be duplicated elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Repo-root-relative locations.
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "nse" / "db" / "nse.sqlite3"
SCHEMA_PATH = ROOT / "nse" / "db" / "schema.sql"
SANDBOX_REPOS = ROOT / "nse" / "sandbox_repos"


@dataclass(frozen=True)
class Hyperparams:
    """Decision-rule constants. Frozen: the arbiter relies on these being stable."""

    k: int = 3                       # candidate branches per task
    M: int = 5                       # ensemble heads
    alpha: float = 0.7               # weight on latent vs simulator p_t
    lambda1: float = 1.0             # critic risk penalty
    lambda2: float = 0.5             # long-term architecture risk penalty
    lambda3: float = 0.75            # uncertainty penalty (blueprint 2 value)
    tau_prune: float = 0.4           # score threshold below which we prune
    u_max: float = 0.15              # uncertainty above which we gather evidence
    max_variance: float = 0.25       # theoretical max variance of a [0,1] var

    # Calibration / audit.
    ece_threshold: float = 0.05
    audit_sampling_percent: float = 0.05
    audit_every_n_tasks: int = 100
    calibration_retrain_every: int = 50
    false_negative_alarm: float = 0.02


@dataclass(frozen=True)
class Budgets:
    """Strict per-task compute budgets (blueprint 2 section 4.5)."""

    max_sandbox_executions_per_task: int = 3
    max_branch_depth: int = 2
    max_total_llm_tokens_per_task: int = 12_000
    max_wall_clock_per_task_s: int = 180


@dataclass(frozen=True)
class SandboxConfig:
    """Docker hardening (blueprint 2 section 4.7 — non-negotiable)."""

    # The sandbox runs with --network none, so the image must already contain
    # the test deps (pytest etc.). Build it via nse/docker/Dockerfile.sandbox.
    # Override with NSE_SANDBOX_IMAGE for a per-repo image.
    image: str = field(
        default_factory=lambda: os.environ.get("NSE_SANDBOX_IMAGE", "nse-sandbox:latest")
    )
    network_mode: str = "none"
    mem_limit: str = "2g"
    cpu_quota: int = 200_000          # 2 CPUs at default 100_000 period
    user: str = "1000:1000"
    timeout_s: int = 60
    # Paths a branch is forbidden to touch; touching them auto-prunes.
    sensitive_paths: tuple[str, ...] = (
        ".env",
        ".env.local",
        "credentials.json",
        "id_rsa",
        ".aws/credentials",
        ".git/config",
    )
    # Full-file rewrite is only allowed under these limits (blueprint 4.1).
    full_rewrite_max_lines: int = 300
    max_changed_files_fraction: float = 0.30
    # The fraction guard only engages once at least this many files are
    # touched, so a legitimate single-file edit is never flagged as "sweeping".
    max_changed_files_min_abs: int = 3


@dataclass(frozen=True)
class LLMConfig:
    """vLLM / OpenAI-compatible inference host settings.

    vLLM has no native Windows wheels, so ``base_url`` defaults to the local
    mock server (see ``nse/scripts/vllm_mock.py``).
    """

    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "llama-3-q8"
    request_timeout_s: float = 60.0
    max_retries: int = 2
    planner_temperature: float = 0.8
    simulator_temperature: float = 0.2
    critic_temperature: float = 1.0
    max_output_tokens: int = 1024


@dataclass(frozen=True)
class Settings:
    hp: Hyperparams = field(default_factory=Hyperparams)
    budgets: Budgets = field(default_factory=Budgets)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


SETTINGS = Settings()
