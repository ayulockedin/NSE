"""Coverage-as-uncertainty signal (Phase 7.1).

A patch whose edited lines aren't exercised by the test suite yields a
*low-evidence* "pass" — the tests can't catch a regression there. We measure
line coverage of the **pristine** (trusted) repo and treat poorly-covered
changes as uncertainty, so the arbiter routes them to gather more evidence
(``INCREMENTAL_SANDBOX``) instead of trusting the green run.

Because coverage is measured on the *pristine* tree, we map a patch to the
**old-file line numbers** it touches (the lines being replaced/removed, or the
surrounding context for pure insertions) — those are the lines whose existing
coverage tells us whether a regression in that region would be caught.

``coverage.py`` is optional: when unavailable the signal degrades to ``0.0`` and
the pipeline behaves exactly as before.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import coverage  # noqa: F401

    _COVERAGE = True
except ImportError:  # pragma: no cover
    _COVERAGE = False


def coverage_available() -> bool:
    return _COVERAGE


@dataclass(frozen=True)
class FileCoverage:
    """Per-file coverage. ``executable`` = ``executed | missing`` so we can ignore
    non-executable changed lines (comments, blanks, docstrings), which coverage
    never marks as covered and would otherwise look falsely under-tested."""

    executed: set[int]
    executable: set[int]


# ───────────────────────────── measurement ──────────────────────────────


def measure_line_coverage(
    repo: Path | str, targets: list[str] | None = None, timeout: int = 120
) -> dict[str, FileCoverage] | None:
    """Run the repo's tests under coverage; return ``{relpath: covered_lines}``.

    Returns ``None`` when coverage.py is unavailable or the run/parse fails — the
    caller treats that as "no coverage signal" (uncertainty contribution 0).
    Measured on the trusted pristine repo, so it runs on the host (no sandbox).
    """
    if not _COVERAGE:
        return None
    repo = Path(repo)
    targets = targets or ["."]
    with tempfile.TemporaryDirectory(prefix="nse_cov_") as td:
        data_file = Path(td) / ".coverage"
        try:
            subprocess.run(
                [sys.executable, "-m", "coverage", "run",
                 "--data-file", str(data_file), f"--source={repo}",
                 "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets],
                cwd=repo, capture_output=True, text=True, timeout=timeout,
            )
            proc = subprocess.run(
                [sys.executable, "-m", "coverage", "json",
                 "--data-file", str(data_file), "-o", "-"],
                cwd=repo, capture_output=True, text=True, timeout=timeout,
            )
            data = json.loads(proc.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError, ValueError):
            return None

    out: dict[str, FileCoverage] = {}
    for path, info in data.get("files", {}).items():
        key = Path(path)
        if key.is_absolute():
            try:
                key = key.relative_to(repo)
            except ValueError:
                pass
        executed = set(info.get("executed_lines", []))
        missing = set(info.get("missing_lines", []))
        out[key.as_posix()] = FileCoverage(executed=executed, executable=executed | missing)
    return out


# ──────────────────────────── change parsing ────────────────────────────


def changed_lines_from_diff(patch_text: str) -> dict[str, set[int]]:
    """Map each file in a unified diff to the **old-file** line numbers of the
    region it changes: the removed (``-``) lines, or — for a pure insertion with
    no removals — the hunk's context lines as a locality proxy."""
    removed: dict[str, set[int]] = {}
    context: dict[str, set[int]] = {}
    cur: str | None = None
    old_ln = 0
    for line in patch_text.splitlines():
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            p = line[4:].strip()
            cur = p[2:] if p.startswith("b/") else p
            continue
        if line.startswith("@@"):
            m = re.search(r"-(\d+)", line)
            old_ln = int(m.group(1)) if m else 0
            continue
        if cur is None:
            continue
        if line.startswith("+"):
            continue  # added line: no old-file number
        if line.startswith("-"):
            removed.setdefault(cur, set()).add(old_ln)
            old_ln += 1
        else:  # context line (incl. blank " ")
            context.setdefault(cur, set()).add(old_ln)
            old_ln += 1

    return {f: (removed.get(f) or context.get(f, set())) for f in removed | context}


def changed_lines(
    patch_text: str | None,
    full_rewrites: dict[str, str] | None,
    pristine_line_counts: dict[str, int] | None = None,
) -> dict[str, set[int]]:
    """Old-file changed lines for a branch (diff and/or full-file rewrites).

    A full-file rewrite touches the whole file, so every pristine line counts
    (``pristine_line_counts`` supplies each file's length)."""
    out: dict[str, set[int]] = {}
    if patch_text:
        out.update(changed_lines_from_diff(patch_text))
    for f in full_rewrites or {}:
        n = (pristine_line_counts or {}).get(f, 0)
        if n:
            out[f] = set(range(1, n + 1))
    return out


# ─────────────────────────── uncertainty map ────────────────────────────


def coverage_fraction(
    coverage_map: dict[str, FileCoverage] | None, changed: dict[str, set[int]]
) -> float | None:
    """Fraction of the *executable* changed lines that the pristine tests cover.

    Non-executable changed lines (comments/blanks/docstrings) are excluded — they
    can never be "covered" and would otherwise look falsely under-tested. Returns
    None when there's no coverage data or no executable changed line to judge."""
    if coverage_map is None:
        return None
    total = 0
    covered = 0
    for f, lines in changed.items():
        fc = coverage_map.get(f)
        if fc is None:
            continue
        executable = lines & fc.executable
        total += len(executable)
        covered += len(executable & fc.executed)
    if total == 0:
        return None
    return covered / total


def coverage_uncertainty(
    coverage_map: dict[str, FileCoverage] | None,
    changed: dict[str, set[int]],
    full_scale: float,
) -> float:
    """Uncertainty contribution in ``[0, full_scale]``: 0 when fully covered (or
    unknown), ``full_scale`` when 0% of the changed lines are covered."""
    frac = coverage_fraction(coverage_map, changed)
    if frac is None:
        return 0.0
    return (1.0 - frac) * full_scale
