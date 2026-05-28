"""Robust patch application — triple-tier fallback (blueprint section 4.1).

Tier 1: ``git apply --index`` (preferred, exact).
Tier 2: ``git apply --3way`` then fuzzy apply via diff-match-patch.
Tier 3: full-file rewrite, only when the Planner supplied ``full_file_rewrites``
        and the file is under the size cap.

Safety: if a patch touches a sensitive path or more than the allowed fraction
of files, the branch is marked unsafe and not applied.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from nse.config import SETTINGS

try:  # diff-match-patch is optional at import time
    from diff_match_patch import diff_match_patch
except ImportError:  # pragma: no cover
    diff_match_patch = None  # type: ignore[assignment]


class PatchApplyError(RuntimeError):
    """All patch application tiers failed."""


class PatchSafetyError(RuntimeError):
    """Patch violates a safety policy (sensitive path / footprint)."""


_DIFF_FILE_RE = re.compile(r"^\+\+\+ [ab]/(.+)$", re.MULTILINE)
_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "venv"}


def _count_source_files(repo_dir: Path) -> int:
    return sum(
        1
        for p in repo_dir.rglob("*")
        if p.is_file() and not _IGNORED_DIRS.intersection(p.parts)
    )


def _files_in_diff(patch_text: str) -> list[str]:
    return [m.group(1).strip() for m in _DIFF_FILE_RE.finditer(patch_text)]


def _is_sensitive(path: str) -> bool:
    norm = path.replace("\\", "/").lower()
    return any(norm.endswith(s) or f"/{s}" in norm for s in SETTINGS.sandbox.sensitive_paths)


def assert_patch_safe(
    repo_dir: Path,
    touched_files: list[str],
) -> None:
    """Raise :class:`PatchSafetyError` if the change footprint is disallowed."""
    for f in touched_files:
        if _is_sensitive(f):
            raise PatchSafetyError(f"patch touches sensitive path: {f}")

    n_touched = len(touched_files)
    if n_touched < SETTINGS.sandbox.max_changed_files_min_abs:
        return  # too few files to count as a sweeping change
    total = _count_source_files(repo_dir)
    if total and n_touched / total > SETTINGS.sandbox.max_changed_files_fraction:
        raise PatchSafetyError(
            f"patch touches {n_touched}/{total} files "
            f"(> {SETTINGS.sandbox.max_changed_files_fraction:.0%})"
        )


def _git(args: list[str], cwd: Path, stdin: Optional[bytes] = None) -> int:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
    )
    return proc.returncode


def _ensure_git_repo(repo_dir: Path) -> None:
    if not (repo_dir / ".git").exists():
        _git(["init", "-q"], repo_dir)
        _git(["add", "-A"], repo_dir)
        _git(["-c", "user.email=nse@local", "-c", "user.name=nse",
              "commit", "-q", "-m", "snapshot", "--allow-empty"], repo_dir)


def _git_apply(repo_dir: Path, patch_text: str, three_way: bool = False) -> bool:
    args = ["apply", "--index"]
    if three_way:
        args = ["apply", "--3way"]
    return _git(args, repo_dir, stdin=patch_text.encode()) == 0


def _fuzzy_apply(repo_dir: Path, patch_text: str) -> bool:
    """Last-ditch hunk alignment using diff-match-patch.

    Only handles simple single-file unified diffs. Returns False if it cannot
    confidently apply every hunk.
    """
    if diff_match_patch is None:
        return False
    files = _files_in_diff(patch_text)
    if len(files) != 1:
        return False
    target = repo_dir / files[0]
    if not target.exists():
        return False

    body_lines = [
        ln[1:] if ln and ln[0] in "+ " else ""
        for ln in patch_text.splitlines()
        if not ln.startswith(("---", "+++", "@@", "diff", "index"))
    ]
    new_text = "\n".join(body_lines)
    dmp = diff_match_patch()
    original = target.read_text(encoding="utf-8")
    patches = dmp.patch_make(original, new_text)
    result, applied = dmp.patch_apply(patches, original)
    if not all(applied):
        return False
    target.write_text(result, encoding="utf-8")
    return True


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def apply_patch_strict_then_fallback(
    repo_dir: Path | str,
    patch_text: Optional[str] = None,
    full_rewrites: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Apply a patch with the three-tier strategy.

    Returns ``{"mode": <tier>}`` on success. Raises :class:`PatchSafetyError`
    or :class:`PatchApplyError` on failure.
    """
    repo_dir = Path(repo_dir)
    touched = _files_in_diff(patch_text or "")
    touched += list((full_rewrites or {}).keys())
    assert_patch_safe(repo_dir, touched)

    if patch_text:
        _ensure_git_repo(repo_dir)
        # Tier 1
        if _git_apply(repo_dir, patch_text):
            return {"mode": "git_apply"}
        # Tier 2a
        if _git_apply(repo_dir, patch_text, three_way=True):
            return {"mode": "git_3way_apply"}
        # Tier 2b
        if _fuzzy_apply(repo_dir, patch_text):
            return {"mode": "fuzzy_apply"}

    # Tier 3: full rewrite (size-limited)
    if full_rewrites:
        for rel, text in full_rewrites.items():
            n_lines = len(text.splitlines())
            if n_lines >= SETTINGS.sandbox.full_rewrite_max_lines:
                raise PatchSafetyError(
                    f"full rewrite of {rel} has {n_lines} lines "
                    f"(>= {SETTINGS.sandbox.full_rewrite_max_lines})"
                )
            _atomic_write(repo_dir / rel, text)
        return {"mode": "full_rewrite"}

    raise PatchApplyError("all patch application methods failed")
