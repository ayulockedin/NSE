"""Real-defect mining from git history (Phase 11.1 + 11.3).

The latent model was trained on synthetic AST mutants. Phase 9 exposed the
*toy->real* gap: real LLM patches get under-scored and pruned because the model
never saw a real defect. This module closes that gap by harvesting **verified
real bug-fix pairs** straight from a repository's git history, in the exact same
:class:`~nse.data.dataset.LabeledExample` shape the seed dataset and the
orchestrator use (so there is no train/serve skew).

Protocol (SWE-bench style — the *fix-added* test is the oracle)
---------------------------------------------------------------
For a candidate fix commit ``C`` with parent ``P`` we split its changed Python
files into **source** ``S`` (the code fix) and **tests** ``T``:

* **fixed tree**  = the repo exactly at ``C``.
* **buggy tree**  = ``C`` but with every source file in ``S`` reverted to ``P``.
  Crucially the *tests stay at ``C``* — so a regression test the fix commit adds
  is present while the code is still broken, which is what makes the bug visible.

We then **verify by execution** (never assume the label): run both trees through
the hardened sandbox and keep the pair only if the buggy tree **fails** and the
fixed tree **passes**. That single rule filters out flaky tests, dependency
drift, and commits that don't actually fix a test-visible defect.

Each verified pair yields two ground-truth examples sharing the changed
function's CPG context:

* the **fix** (``S`` diff ``P -> C``), label ``1`` — the real patch the toy model
  under-scores; this is the example that matters most, so it is up-weighted.
* the **regression** (``C -> P``), label ``0`` — a real breaking change.

Measured ``r_long`` (Phase 11.3) replaces the structural blast-radius proxy with
a *temporal* one: how often the fixed function is reworked again in the commits
that follow — a fix that has to be re-patched is genuinely fragile.

Everything here reads git **read-only** (``log``/``show``/``archive``/``diff``)
and materialises historical trees into throwaway temp dirs via ``git archive``,
so the caller's working tree is never touched.
"""

from __future__ import annotations

import ast
import io
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from nse.data.dataset import (
    LabeledExample,
    _complexity,
    _unified_diff,
)
from nse.memory_graph.graph_db import MemoryGraph
from nse.models.cpg_features import (
    blast_radius,
    build_cpg_features,
    changed_function_names,
)
from nse.models.latent_model import LatentEnsemble
from nse.orchestrator.executor import run_sandbox
from nse.orchestrator.schemas import PlannerBranch

# Subjects that usually denote a bug fix — a cheap prefilter so we don't sandbox
# every commit. Verification (buggy-fails ∧ fixed-passes) is the real gate; this
# only narrows the candidate set. Override via ``message_regex``.
DEFAULT_FIX_REGEX = r"\b(fix(e[ds])?|bug|defect|regression|issue|crash|broken|incorrect)\b"

# Real examples outweigh synthetic mutants during curriculum fine-tuning.
DEFAULT_REAL_WEIGHT = 3.0


@dataclass
class MinedPair:
    """A candidate (parent -> fix) commit pair with its changed Python files."""

    sha: str
    parent: str
    subject: str
    source_files: list[str]   # non-test .py changed by the fix (the code patch)
    test_files: list[str]     # test .py changed by the fix (kept at the fix rev)


# ───────────────────────────── git plumbing ─────────────────────────────


def _git(repo: Path | str, *args: str) -> str:
    """Run a read-only git command and return stdout (text)."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, ["git", *args], proc.stdout, proc.stderr
        )
    return proc.stdout


def _git_ok(repo: Path | str, *args: str) -> str | None:
    """Like :func:`_git` but returns ``None`` instead of raising (for paths that
    may not exist at a given revision)."""
    try:
        return _git(repo, *args)
    except subprocess.CalledProcessError:
        return None


def _materialize_tree(repo: Path | str, sha: str, dest: Path) -> None:
    """Extract the full repo tree at ``sha`` into ``dest`` via ``git archive``.

    No checkout, no working-tree disturbance, cross-platform (we untar in-process
    rather than shelling out to ``tar``). The archive content is the user's own
    repository, so in-process extraction is trusted.
    """
    raw = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", sha],
        capture_output=True,
        check=True,
    ).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        # ``filter="data"`` rejects path-traversal / unsafe members and is the
        # default from Python 3.14; setting it explicitly silences the 3.13
        # deprecation warning and hardens extraction even on a trusted archive.
        tf.extractall(dest, filter="data")


def _file_at(repo: Path | str, sha: str, rel: str) -> str | None:
    """Contents of ``rel`` at revision ``sha``, or ``None`` if absent there."""
    return _git_ok(repo, "show", f"{sha}:{rel}")


def is_test_file(rel: str) -> bool:
    """Heuristic: pytest-style test modules and anything under a tests/ dir."""
    p = Path(rel)
    if any(part in {"tests", "test"} for part in p.parts):
        return True
    return p.name.startswith("test_") or p.stem.endswith("_test")


def _parent_of(repo: Path | str, sha: str) -> str | None:
    """Sole parent of ``sha``; ``None`` for a root commit or a merge (we only
    mine clean single-parent fixes, where the bug<->fix delta is unambiguous)."""
    parents = _git_ok(repo, "log", "--format=%P", "-n", "1", sha)
    if not parents:
        return None
    shas = parents.split()
    if len(shas) != 1:  # root (0) or merge (>1)
        return None
    return shas[0]


# ──────────────────────────── candidate finding ────────────────────────────


def find_fix_commits(
    repo: Path | str,
    limit: int = 200,
    rev: str = "HEAD",
    message_regex: str | None = DEFAULT_FIX_REGEX,
) -> list[str]:
    """Return up to ``limit`` candidate fix-commit SHAs (newest first).

    Filters to non-merge commits whose subject matches ``message_regex`` (set it
    to ``None`` to consider every commit). This is only a prefilter; whether a
    commit is a *real* defect fix is decided later by execution.
    """
    args = ["log", "--no-merges", "--format=%H%x00%s", rev]
    out = _git(repo, *args)
    pat = None
    if message_regex:
        import re

        pat = re.compile(message_regex, re.IGNORECASE)

    shas: list[str] = []
    for line in out.splitlines():
        if "\x00" not in line:
            continue
        sha, subject = line.split("\x00", 1)
        if pat is None or pat.search(subject):
            shas.append(sha)
        if len(shas) >= limit:
            break
    return shas


def extract_pair(repo: Path | str, sha: str) -> MinedPair | None:
    """Build a :class:`MinedPair` for ``sha``, or ``None`` if it is not a
    single-parent commit that changes at least one non-test Python source file."""
    parent = _parent_of(repo, sha)
    if parent is None:
        return None
    names = _git_ok(repo, "diff", "--name-only", parent, sha)
    if not names:
        return None
    py = [f for f in names.splitlines() if f.endswith(".py")]
    sources = [f for f in py if not is_test_file(f)]
    tests = [f for f in py if is_test_file(f)]
    if not sources:
        return None
    subject = (_git_ok(repo, "log", "--format=%s", "-n", "1", sha) or "").strip()
    return MinedPair(
        sha=sha, parent=parent, subject=subject, source_files=sources, test_files=tests
    )


# ──────────────────────────── tree construction ────────────────────────────


def _build_buggy_tree(repo: Path | str, pair: MinedPair, fixed_dir: Path, dest: Path) -> None:
    """Copy the fixed tree, then revert only the *source* files to the parent
    revision — leaving the (possibly newly-added) tests in place so the defect is
    still observable."""
    shutil.copytree(fixed_dir, dest)
    for rel in pair.source_files:
        target = dest / rel
        parent_src = _file_at(repo, pair.parent, rel)
        if parent_src is None:
            # File was added by the fix; it must not exist in the buggy tree.
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(parent_src, encoding="utf-8")


# ─────────────────────────── measured r_long (11.3) ───────────────────────────


def _func_segment(source: str, fn: str) -> str | None:
    """Exact source segment of top-or-nested function ``fn`` in ``source``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn:
            return ast.get_source_segment(source, node)
    return None


def measured_r_long(
    repo: Path | str,
    sha: str,
    rel: str,
    fn: str,
    horizon: int = 10,
    norm: float = 3.0,
) -> float:
    """Temporal fragility in ``[0, 1]``: how often function ``fn`` in ``rel`` is
    reworked in the commits that follow ``sha``.

    We walk the next ``horizon`` commits that touch ``rel`` (oldest first) and
    count those that actually change ``fn``'s source — a fix that has to be
    re-patched is fragile / high long-term risk. Falls back to ``0.0`` if the
    history can't be read. This is the *measured* replacement for the structural
    :func:`blast_radius` proxy.
    """
    if not fn:
        return 0.0
    later = _git_ok(repo, "log", "--reverse", "--format=%H", f"{sha}..HEAD", "--", rel)
    if not later:
        return 0.0
    reworks = 0
    seen = 0
    for csha in later.splitlines():
        if seen >= horizon:
            break
        seen += 1
        cparent = _parent_of(repo, csha)
        if cparent is None:
            continue
        before = _file_at(repo, cparent, rel)
        after = _file_at(repo, csha, rel)
        if before is None or after is None:
            continue
        if _func_segment(before, fn) != _func_segment(after, fn):
            reworks += 1
    return min(1.0, reworks / norm)


# ──────────────────────────── example assembly ────────────────────────────


def _make_example(
    *,
    diff: str,
    edited_files: list[str],
    label: int,
    mg: MemoryGraph,
    center_functions: list[str],
    kind: str,
    description: str,
    r_long_target: float,
    weight: float,
) -> LabeledExample:
    """Assemble one :class:`LabeledExample`, mirroring the dataset builder's
    feature pipeline exactly (patch features + per-function CPG subgraph)."""
    branch = PlannerBranch(
        branch_id=str(uuid.uuid4()),
        strategy=description,
        edited_files=edited_files,
        patch_preview=diff,
        expected_complexity=_complexity(diff),
        planner_confidence=0.5,
    )
    feats = LatentEnsemble.patch_features(branch)
    node_features, edge_index = build_cpg_features(
        mg, edited_files, center_functions=center_functions or None
    )
    return LabeledExample(
        features=feats,
        label=label,
        assumed_breaking=(label == 0),
        kind=kind,
        description=description,
        file=edited_files[0],
        sandbox_mode="git_mined",
        runtime=0.0,
        node_features=node_features,
        edge_index=edge_index,
        r_long_target=r_long_target,
        weight=weight,
    )


def _concat_diff(buggy_dir: Path, fixed_dir: Path, files: list[str], reverse: bool) -> str:
    """Unified diff over ``files`` between the two trees. ``reverse`` swaps the
    direction (fix = buggy->fixed; regression = fixed->buggy)."""
    chunks: list[str] = []
    for rel in files:
        b = (buggy_dir / rel).read_text(encoding="utf-8") if (buggy_dir / rel).exists() else ""
        f = (fixed_dir / rel).read_text(encoding="utf-8") if (fixed_dir / rel).exists() else ""
        old, new = (f, b) if reverse else (b, f)
        chunks.append(_unified_diff(old, new, rel))
    return "".join(chunks)


def build_real_examples(
    repo: Path | str,
    limit: int = 200,
    max_pairs: int | None = 25,
    force_local: bool = False,
    full: bool = True,
    real_weight: float = DEFAULT_REAL_WEIGHT,
    r_long_horizon: int = 10,
    message_regex: str | None = DEFAULT_FIX_REGEX,
    test_scope: list[str] | None = None,
    verbose: bool = False,
) -> list[LabeledExample]:
    """Mine ``repo``'s history into verified real-defect training examples.

    Scans up to ``limit`` candidate commits and keeps the first ``max_pairs``
    whose buggy/fixed trees the sandbox confirms fail/pass respectively. Returns
    two examples (fix + regression) per verified pair.

    ``force_local=True`` skips Docker (CI / Windows-without-Docker); for *trusted*
    labels run with Docker available. ``full=True`` runs the whole suite — the
    honest definition of "fixed = green" for an arbitrary repo whose test->file
    map we don't know. ``test_scope`` pins the verification run to specific test
    paths (e.g. ``["tests"]``) — use it when a repo's full suite has env-broken
    peripheral tests, or to scope to the relevant module for speed.
    """
    repo = Path(repo)
    examples: list[LabeledExample] = []
    verified = 0

    for sha in find_fix_commits(repo, limit=limit, message_regex=message_regex):
        if max_pairs is not None and verified >= max_pairs:
            break
        pair = extract_pair(repo, sha)
        if pair is None:
            continue

        work = Path(tempfile.mkdtemp(prefix="nse_mine_"))
        fixed_dir = work / "fixed"
        buggy_dir = work / "buggy"
        try:
            _materialize_tree(repo, pair.sha, fixed_dir)
            _build_buggy_tree(repo, pair, fixed_dir, buggy_dir)

            buggy_run = run_sandbox(
                buggy_dir, edited_files=pair.source_files, full=full,
                force_local=force_local, pytest_targets=test_scope,
            )
            fixed_run = run_sandbox(
                fixed_dir, edited_files=pair.source_files, full=full,
                force_local=force_local, pytest_targets=test_scope,
            )
            # Verify by execution: the fix must turn the suite red -> green.
            if buggy_run.tests_passed or not fixed_run.tests_passed:
                if verbose:
                    print(
                        f"  skip {sha[:8]} '{pair.subject[:48]}' "
                        f"(buggy_pass={buggy_run.tests_passed} fixed_pass={fixed_run.tests_passed})"
                    )
                continue

            fns = changed_function_names(buggy_dir, fixed_dir, pair.source_files)
            mg_buggy = MemoryGraph(buggy_dir).build_graph()
            mg_fixed = MemoryGraph(fixed_dir).build_graph()
            r_long = (
                measured_r_long(repo, pair.sha, pair.source_files[0], fns[0], r_long_horizon)
                if fns
                else _structural_r_long(mg_fixed, pair.source_files[0], fns)
            )

            fix_diff = _concat_diff(buggy_dir, fixed_dir, pair.source_files, reverse=False)
            reg_diff = _concat_diff(buggy_dir, fixed_dir, pair.source_files, reverse=True)
            # Lead the description with the short SHA so the two examples of a
            # pair can be regrouped into a defect "task" downstream (% resolved).
            tag = f"{pair.sha[:8]} {pair.subject[:72]}"
            examples.append(
                _make_example(
                    diff=fix_diff,
                    edited_files=pair.source_files,
                    label=1,
                    mg=mg_buggy,  # graph reflects pre-patch (buggy) state, as at serve time
                    center_functions=fns,
                    kind="real_fix",
                    description=tag,
                    r_long_target=r_long,
                    weight=real_weight,
                )
            )
            examples.append(
                _make_example(
                    diff=reg_diff,
                    edited_files=pair.source_files,
                    label=0,
                    mg=mg_fixed,  # regression is applied to the currently-good tree
                    center_functions=fns,
                    kind="real_regression",
                    description=tag,
                    r_long_target=r_long,
                    weight=real_weight,
                )
            )
            verified += 1
            if verbose:
                print(f"  + pair {sha[:8]} '{pair.subject[:48]}' fns={fns}")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    return examples


def _structural_r_long(mg: MemoryGraph, rel: str, fns: list[str]) -> float:
    """Fallback to the structural blast-radius proxy when no function could be
    attributed (so the r_long head still gets a defined, bounded target)."""
    if not fns:
        return 0.0
    return blast_radius(mg, f"{rel}::{fns[0]}")


# ────────────────────────────────── CLI ────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse

    from nse.data.dataset import write_jsonl

    parser = argparse.ArgumentParser(description="Mine verified real bug-fix pairs from a git repo.")
    parser.add_argument("--repo", required=True, type=Path, help="path to a git repository")
    parser.add_argument("--out", required=True, type=Path, help="output JSONL path")
    parser.add_argument("--limit", type=int, default=200, help="max candidate commits to scan")
    parser.add_argument("--max-pairs", type=int, default=25, help="stop after N verified pairs")
    parser.add_argument("--force-local", action="store_true", help="label without Docker (unsafe)")
    parser.add_argument(
        "--no-message-filter",
        action="store_true",
        help="consider every commit, not just fix-worded subjects",
    )
    parser.add_argument(
        "--test-scope",
        nargs="+",
        default=None,
        metavar="PATH",
        help="pin verification to these test paths (e.g. tests) instead of the full suite",
    )
    args = parser.parse_args(argv)

    print(f"Mining {args.repo} (scan<= {args.limit}, target {args.max_pairs} pairs) ...")
    examples = build_real_examples(
        args.repo,
        limit=args.limit,
        max_pairs=args.max_pairs,
        force_local=args.force_local,
        message_regex=None if args.no_message_filter else DEFAULT_FIX_REGEX,
        test_scope=args.test_scope,
        verbose=True,
    )
    passed = sum(e.label for e in examples)
    print(
        f"{len(examples)} examples from {len(examples) // 2} verified pairs "
        f"({passed} fix / {len(examples) - passed} regression)"
    )
    out = write_jsonl(examples, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
