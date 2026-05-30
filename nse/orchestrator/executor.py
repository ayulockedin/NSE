"""Sandbox Executor — isolated patch execution + incremental verification.

Primary path: Docker (network disabled, all caps dropped, no-new-privileges,
non-root, read-only base FS, resource-limited) per the non-negotiable security
rules (blueprint 4.7).

Fallback path: a local subprocess runner used ONLY when Docker is unavailable
(e.g. Windows dev without Docker Desktop). It is explicitly insecure and must
never run untrusted patches — it exists so the rest of the pipeline can be
exercised end-to-end during development.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nse.config import SETTINGS
from nse.orchestrator.schemas import Outcome

try:
    import docker  # type: ignore
    from docker.errors import DockerException  # type: ignore
except ImportError:  # pragma: no cover
    docker = None  # type: ignore
    DockerException = Exception  # type: ignore


@dataclass
class SandboxRun:
    compiled: bool
    tests_passed: bool
    logs: str
    runtime: float
    mode: str  # "docker" | "local_unsafe"
    # Richer signal (Phase 8.2), parsed from pytest output. Counts are capped by
    # --maxfail=1, so n_failed is 0 or 1 and failed_tests holds the test that
    # caught the regression.
    n_passed: int = 0
    n_failed: int = 0
    failed_tests: list[str] = field(default_factory=list)


def _parse_pytest_summary(output: str) -> tuple[int, int, list[str]]:
    """Best-effort parse of pytest output -> (n_passed, n_failed, failed_ids)."""
    n_passed = n_failed = 0
    m = re.search(r"(\d+)\s+passed", output)
    if m:
        n_passed = int(m.group(1))
    m = re.search(r"(\d+)\s+(?:failed|error)", output)
    if m:
        n_failed = int(m.group(1))
    failed = re.findall(r"^(?:FAILED|ERROR)\s+(\S+)", output, re.MULTILINE)
    return n_passed, n_failed, failed


def docker_available() -> bool:
    if docker is None:
        return False
    try:
        docker.from_env().ping()
        return True
    except DockerException:
        return False


def _affected_tests(edited_files: list[str], tests_map: dict[str, list[str]]) -> set[str]:
    affected: set[str] = set()
    for f in edited_files:
        affected.update(tests_map.get(f, []))
    return affected


def _pytest_cmd(affected: set[str]) -> str:
    # ``--rootdir .`` + an explicit target pins pytest to this repo so it can't
    # inherit an enclosing project's config (rootdir / testpaths) when the repo
    # happens to be nested under one.
    targets = " ".join(sorted(affected)) if affected else "."
    # -rfE surfaces FAILED/ERROR test ids in the summary so we can capture which
    # test caught a regression (Phase 8.2).
    return f"pytest {targets} --rootdir . -p no:cacheprovider --maxfail=1 -q -rfE"


# ─────────────────────────────── Docker path ───────────────────────────────


def _run_docker(repo_dir: Path, build_cmd: str, timeout: int) -> SandboxRun:
    client = docker.from_env()
    cfg = SETTINGS.sandbox
    start = time.time()
    container = client.containers.run(
        image=cfg.image,
        command=["bash", "-lc", build_cmd],
        working_dir="/workspace/repo",
        volumes={str(repo_dir): {"bind": "/workspace/repo", "mode": "rw"}},
        network_mode=cfg.network_mode,
        cap_drop=["ALL"],
        security_opt=["no-new-privileges"],
        read_only=True,
        tmpfs={"/tmp": ""},
        detach=True,
        user=cfg.user,
        mem_limit=cfg.mem_limit,
        cpu_quota=cfg.cpu_quota,
    )
    try:
        result = container.wait(timeout=timeout)
        logs = container.logs().decode(errors="replace")
        exit_code = int(result.get("StatusCode", 1))
    finally:
        container.remove(force=True)
    runtime = time.time() - start
    n_passed, n_failed, failed = _parse_pytest_summary(logs)
    return SandboxRun(
        compiled=True,                 # symbolic gate already proved compilation
        tests_passed=exit_code == 0,
        logs=logs,
        runtime=runtime,
        mode="docker",
        n_passed=n_passed,
        n_failed=n_failed,
        failed_tests=failed,
    )


# ──────────────────────────── Local fallback path ───────────────────────────


def _run_local_unsafe(repo_dir: Path, build_cmd: str, timeout: int) -> SandboxRun:
    """DEV ONLY. No isolation. Never use for untrusted code."""
    start = time.time()
    # Translate the pytest invocation to the current interpreter.
    cmd = build_cmd.replace("pytest", f"{sys.executable} -m pytest", 1)
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_dir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        logs = proc.stdout + proc.stderr
        passed = proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        logs = f"timeout after {timeout}s\n{exc.stdout or ''}"
        passed = False
    n_passed, n_failed, failed = _parse_pytest_summary(logs)
    return SandboxRun(
        compiled=True,
        tests_passed=passed,
        logs=logs,
        runtime=time.time() - start,
        mode="local_unsafe",
        n_passed=n_passed,
        n_failed=n_failed,
        failed_tests=failed,
    )


# ─────────────────────────────── Public API ────────────────────────────────


def run_sandbox(
    repo_snapshot_path: Path | str,
    edited_files: Optional[list[str]] = None,
    tests_map: Optional[dict[str, list[str]]] = None,
    full: bool = False,
    timeout: Optional[int] = None,
    force_local: bool = False,
    pytest_targets: Optional[list[str]] = None,
) -> SandboxRun:
    """Run tests for a patched snapshot in an isolated environment.

    The snapshot is copied to a scratch dir first so the original is untouched.
    Incremental by default: only tests mapped to ``edited_files`` run unless
    ``full=True`` or no mapping is available.

    ``pytest_targets`` pins the run to explicit paths/node-ids (overriding both
    ``full`` and the ``tests_map``) — a *designated test set*. Used when mining a
    repo whose full suite has env-broken peripheral tests, or to scope to the one
    module that exercises a change (faster, SWE-bench style).
    """
    repo_snapshot_path = Path(repo_snapshot_path)
    timeout = timeout or SETTINGS.sandbox.timeout_s
    edited_files = edited_files or []
    tests_map = tests_map or {}

    if pytest_targets is not None:
        affected = set(pytest_targets)
    else:
        affected = set() if full else _affected_tests(edited_files, tests_map)
    build_cmd = _pytest_cmd(affected)

    workdir = Path(tempfile.mkdtemp(prefix="nse_"))
    repo_copy = workdir / "repo"
    try:
        shutil.copytree(repo_snapshot_path, repo_copy)
        if not force_local and docker_available():
            return _run_docker(repo_copy, build_cmd, timeout)
        return _run_local_unsafe(repo_copy, build_cmd, timeout)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def to_outcome(branch_id: str, run: SandboxRun) -> Outcome:
    return Outcome(
        branch_id=branch_id,
        compiled=int(run.compiled),
        tests_passed=int(run.tests_passed),
        logs=run.logs[:20_000],
        runtime=run.runtime,
    )
