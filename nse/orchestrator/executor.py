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

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
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
    return f"pytest {targets} --rootdir . -p no:cacheprovider --maxfail=1 -q"


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
    return SandboxRun(
        compiled=True,                 # symbolic gate already proved compilation
        tests_passed=exit_code == 0,
        logs=logs,
        runtime=runtime,
        mode="docker",
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
    return SandboxRun(
        compiled=True,
        tests_passed=passed,
        logs=logs,
        runtime=time.time() - start,
        mode="local_unsafe",
    )


# ─────────────────────────────── Public API ────────────────────────────────


def run_sandbox(
    repo_snapshot_path: Path | str,
    edited_files: Optional[list[str]] = None,
    tests_map: Optional[dict[str, list[str]]] = None,
    full: bool = False,
    timeout: Optional[int] = None,
    force_local: bool = False,
) -> SandboxRun:
    """Run tests for a patched snapshot in an isolated environment.

    The snapshot is copied to a scratch dir first so the original is untouched.
    Incremental by default: only tests mapped to ``edited_files`` run unless
    ``full=True`` or no mapping is available.
    """
    repo_snapshot_path = Path(repo_snapshot_path)
    timeout = timeout or SETTINGS.sandbox.timeout_s
    edited_files = edited_files or []
    tests_map = tests_map or {}

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
