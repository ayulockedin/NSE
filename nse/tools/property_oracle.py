"""Property-based / metamorphic oracle (Phase 7.3).

When a change is under-tested (Phase 7.1 flags it), we synthesize a safety net by
**differential testing against the pristine function**: run the original and the
patched function on the same fuzzed inputs and look for a *new crash* — the
patched function raising on an input the original handled cleanly. That's
unambiguous breakage regardless of whether the patch was meant to change
behavior, so it's the signal we gate on. (Behavioral *divergence* is recorded too
but not failed on, since a real fix legitimately changes outputs.)

Security: the patched function is untrusted, so :func:`run_property_oracle`
executes it **only inside the Docker sandbox** via a self-contained generated
harness. The in-process :func:`differential_check` is for trusted code (tests)
and is also the logic the harness reimplements inline.
"""

from __future__ import annotations

import ast
import inspect
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nse.orchestrator.executor import SandboxRun, run_sandbox
from nse.orchestrator.patcher import (
    PatchApplyError,
    PatchSafetyError,
    apply_patch_strict_then_fallback,
)

# Varied, type-mixed value pool. Many combinations will error on both functions
# (e.g. str + int) — those are skipped; the informative trials are where the
# original succeeds.
_VALUE_POOL = [
    0, 1, -1, 2, 3, -5, 10, 100,
    0.5, -1.5, 2.0,
    True, False,
    "", "a", "abc", "racecar",
    [], [1, 2, 3], [0], [-1, 5, 2], [1, 1, 1],
]


@dataclass
class OracleResult:
    trials: int
    comparable: int      # inputs the original handled (so the pair is judgeable)
    new_crashes: int     # patched raised where original succeeded
    divergences: int     # both ran, outputs differed
    ok: bool             # no new crashes
    note: str = ""


def generate_inputs(arity: int, n: int = 60, seed: int = 0) -> list[tuple]:
    """Deterministic fuzzed argument tuples of the given arity."""
    import random

    rng = random.Random(seed)
    if arity <= 0:
        return [()]
    seen: set[tuple] = set()
    out: list[tuple] = []
    for _ in range(n * 3):  # oversample then dedup
        args = tuple(rng.choice(_VALUE_POOL) for _ in range(arity))
        key = repr(args)
        if key not in seen:
            seen.add(key)
            out.append(args)
        if len(out) >= n:
            break
    return out


def _equal(a, b) -> bool:
    try:
        if isinstance(a, float) or isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
        return bool(a == b)
    except Exception:
        return False


def _load_func(src: str, name: str):
    """Exec a source string in an isolated namespace and return ``name``.

    TRUSTED CODE ONLY — used host-side on the pristine function and in tests."""
    ns: dict = {}
    exec(compile(src, "<oracle>", "exec"), ns)  # noqa: S102 - trusted source
    return ns.get(name)


def _arity(fn) -> int:
    try:
        params = inspect.signature(fn).parameters.values()
        return sum(
            1 for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
    except (TypeError, ValueError):
        return 0


def differential_check(
    golden_src: str, candidate_src: str, func_name: str, n: int = 60, seed: int = 0
) -> OracleResult:
    """In-process differential test of ``func_name`` (golden vs candidate).

    For trusted code only (tests / inside the sandbox harness)."""
    golden = _load_func(golden_src, func_name)
    candidate = _load_func(candidate_src, func_name)
    if golden is None or candidate is None:
        return OracleResult(0, 0, 0, 0, ok=True, note="function not found")

    inputs = generate_inputs(_arity(golden), n, seed)
    comparable = new_crashes = divergences = 0
    for args in inputs:
        try:
            g = golden(*args)
        except Exception:
            continue  # original can't handle it -> not our concern
        comparable += 1
        try:
            c = candidate(*args)
        except Exception:
            new_crashes += 1
            continue
        if not _equal(c, g):
            divergences += 1
    return OracleResult(
        len(inputs), comparable, new_crashes, divergences, ok=(new_crashes == 0)
    )


def golden_source(repo: Path | str, rel_file: str, func_name: str) -> str | None:
    """Extract the pristine source of ``func_name`` from ``rel_file``."""
    path = Path(repo) / rel_file
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return ast.get_source_segment(path.read_text(encoding="utf-8"), node)
    return None


def build_harness(
    golden_src: str, func_name: str, module_name: str, inputs: list[tuple]
) -> str:
    """A self-contained pytest harness (no nse imports) that fails iff the patched
    ``func_name`` crashes on an input the embedded golden handled."""
    return (
        f"{golden_src}\n\n"
        f"import {module_name} as _m\n"
        f"_cand = _m.{func_name}\n"
        f"_golden = {func_name}\n"
        f"_INPUTS = {inputs!r}\n\n"
        "def test_no_new_crashes():\n"
        "    for args in _INPUTS:\n"
        "        try:\n"
        "            _golden(*args)\n"
        "        except Exception:\n"
        "            continue\n"
        "        try:\n"
        "            _cand(*args)\n"
        "        except Exception as e:\n"
        f"            raise AssertionError("
        f"f'patched {func_name} crashed on '"
        " + repr(args) + f' where original succeeded: {e}')\n"
    )


def run_property_oracle(
    pristine_repo: Path | str,
    branch,
    func_name: str,
    golden_src: str,
    n: int = 60,
    seed: int = 0,
    force_local: bool = False,
) -> SandboxRun | None:
    """Securely run the differential harness for one branch.

    Applies the branch's patch to a copy of the pristine repo, drops in a
    generated harness, and runs it in the sandbox. Returns the ``SandboxRun``
    (``tests_passed`` False == a new crash was found), or ``None`` if the patch
    couldn't be applied or inputs couldn't be built.
    """
    if not branch.edited_files:
        return None
    rel_file = branch.edited_files[0]
    golden = _load_func(golden_src, func_name)
    if golden is None:
        return None
    inputs = generate_inputs(_arity(golden), n, seed)
    harness = build_harness(golden_src, func_name, Path(rel_file).stem, inputs)

    work = Path(tempfile.mkdtemp(prefix="nse_oracle_"))
    repo = work / "repo"
    try:
        shutil.copytree(pristine_repo, repo)
        try:
            apply_patch_strict_then_fallback(
                repo,
                patch_text=branch.patch_preview,
                full_rewrites=branch.full_file_rewrites,
            )
        except (PatchApplyError, PatchSafetyError):
            return None
        harness_name = "test_nse_oracle.py"
        (repo / harness_name).write_text(harness, encoding="utf-8")
        # Run ONLY the harness, not the repo's own (possibly failing) tests.
        return run_sandbox(
            repo,
            edited_files=[rel_file],
            tests_map={rel_file: [harness_name]},
            full=False,
            force_local=force_local,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
