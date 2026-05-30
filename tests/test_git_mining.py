"""Tests for real-defect mining from git history (Phase 11.1 + 11.3).

Hermetic: each test builds a throwaway git repo in ``tmp_path`` with controlled
buggy/fixed commits, so the suite is deterministic and offline. Labels come from
the real sandbox in ``force_local`` mode (no Docker needed). No torch required —
mining uses the heuristic featurizer + networkx graph only.
"""

import subprocess

import pytest

from nse.data.git_mining import (
    DEFAULT_REAL_WEIGHT,
    build_real_examples,
    extract_pair,
    find_fix_commits,
    is_test_file,
    measured_r_long,
)

# A clean import sanity-checks the module under the repo's lint/type rules.

CALC_BUGGY = """\
def add(a, b):
    return a - b


def mul(a, b):
    return a * b
"""

CALC_FIXED = """\
def add(a, b):
    return a + b


def mul(a, b):
    return a * b
"""

TEST_CALC = """\
from calc import add, mul


def test_add():
    assert add(2, 3) == 5


def test_mul():
    assert mul(2, 3) == 6
"""


def _git(repo, *args):
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c", "user.name=NSE",
        "-c", "user.email=nse@example.com",
        "-c", "commit.gpgsign=false",
        "commit", "-m", message,
    )
    return _git(repo, "rev-parse", "HEAD")


def _init(repo):
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "core.autocrlf", "false")


def _write(repo, rel, text):
    (repo / rel).write_text(text, encoding="utf-8")


@pytest.fixture
def defect_repo(tmp_path):
    """A repo with a buggy parent and a fix commit that turns the suite green."""
    repo = tmp_path / "proj"
    _init(repo)
    _write(repo, "calc.py", CALC_BUGGY)
    _write(repo, "test_calc.py", TEST_CALC)
    parent = _commit(repo, "initial implementation")
    _write(repo, "calc.py", CALC_FIXED)
    fix = _commit(repo, "Fix add() returning the difference instead of the sum")
    return repo, parent, fix


# ──────────────────────────── unit-level pieces ────────────────────────────


def test_is_test_file():
    assert is_test_file("test_calc.py")
    assert is_test_file("pkg/tests/test_x.py")
    assert is_test_file("foo_test.py")
    assert not is_test_file("calc.py")
    assert not is_test_file("pkg/core.py")


def test_find_fix_commits_filters_by_message(defect_repo):
    repo, _parent, fix = defect_repo
    fixes = find_fix_commits(repo)
    assert fix in fixes
    # The "initial implementation" commit isn't fix-worded -> excluded by filter.
    assert len(fixes) == 1
    # With the filter off, every (non-merge) commit is a candidate.
    assert len(find_fix_commits(repo, message_regex=None)) == 2


def test_extract_pair_separates_source_from_tests(defect_repo):
    repo, parent, fix = defect_repo
    pair = extract_pair(repo, fix)
    assert pair is not None
    assert pair.parent == parent
    assert pair.source_files == ["calc.py"]
    assert pair.test_files == []  # the fix commit didn't touch the test


def test_extract_pair_skips_docs_only_commit(tmp_path):
    repo = tmp_path / "docs"
    _init(repo)
    _write(repo, "calc.py", CALC_FIXED)
    _commit(repo, "initial")
    _write(repo, "README.md", "# hello\n")
    docs = _commit(repo, "Fix typo in docs")
    assert extract_pair(repo, docs) is None  # no .py source changed


# ──────────────────────────── full mining path ────────────────────────────


def test_mines_one_verified_pair(defect_repo):
    repo, _parent, _fix = defect_repo
    examples = build_real_examples(repo, force_local=True, full=True)

    assert len(examples) == 2  # one verified pair -> fix + regression
    by_kind = {e.kind: e for e in examples}
    assert set(by_kind) == {"real_fix", "real_regression"}

    fix_ex = by_kind["real_fix"]
    reg_ex = by_kind["real_regression"]
    assert fix_ex.label == 1 and reg_ex.label == 0
    # Verified by execution, in the exact training schema.
    assert len(fix_ex.features) == 6
    assert fix_ex.node_features  # real per-function CPG subgraph attached
    assert fix_ex.file == "calc.py"
    assert 0.0 <= fix_ex.r_long_target <= 1.0
    assert fix_ex.weight == pytest.approx(DEFAULT_REAL_WEIGHT)
    # Both examples of a pair share the leading SHA tag (for % resolved grouping).
    assert fix_ex.description.split(" ", 1)[0] == reg_ex.description.split(" ", 1)[0]


def test_fix_diff_adds_back_the_sum(defect_repo):
    repo, _parent, _fix = defect_repo
    examples = build_real_examples(repo, force_local=True, full=True)
    fix_ex = next(e for e in examples if e.kind == "real_fix")
    # The fix example's positive class comes from a real green run; its patch
    # feature vector must reflect a non-empty diff (added/removed lines > 0).
    _files, added, removed, *_ = fix_ex.features
    assert added >= 1 and removed >= 1  # one line swapped (a-b -> a+b)


def test_test_scope_pins_verification(defect_repo):
    repo, _parent, _fix = defect_repo
    # Scoping verification to the one relevant test file still finds the pair.
    examples = build_real_examples(repo, force_local=True, test_scope=["test_calc.py"])
    assert len(examples) == 2
    assert {e.kind for e in examples} == {"real_fix", "real_regression"}


def test_unverifiable_pair_is_dropped(tmp_path):
    """A 'fix' that does not actually turn the suite green is rejected."""
    repo = tmp_path / "noop"
    _init(repo)
    _write(repo, "calc.py", CALC_BUGGY)
    _write(repo, "test_calc.py", TEST_CALC)
    _commit(repo, "initial")
    # A cosmetic change that leaves the bug in place -> buggy still fails AND
    # "fixed" still fails -> not a verified pair.
    _write(repo, "calc.py", CALC_BUGGY + "\n\ndef helper():\n    return 0\n")
    _commit(repo, "Fix: refactor (does not address the bug)")
    assert build_real_examples(repo, force_local=True, full=True) == []


# ──────────────────────────── measured r_long (11.3) ────────────────────────


def test_measured_r_long_detects_rework(tmp_path):
    repo = tmp_path / "rework"
    _init(repo)
    _write(repo, "calc.py", CALC_BUGGY)
    _write(repo, "test_calc.py", TEST_CALC)
    _commit(repo, "initial")
    _write(repo, "calc.py", CALC_FIXED)
    fix = _commit(repo, "Fix add() bug")
    # A later commit reworks add() again -> fragile -> r_long > 0.
    _write(repo, "calc.py", CALC_FIXED.replace("return a + b", "return b + a"))
    _commit(repo, "tweak add()")

    assert measured_r_long(repo, fix, "calc.py", "add") > 0.0
    # A function never touched again has zero measured fragility.
    assert measured_r_long(repo, fix, "calc.py", "mul") == 0.0


def test_measured_r_long_zero_without_future_commits(defect_repo):
    repo, _parent, fix = defect_repo
    # 'add' is the fix HEAD; nothing follows it -> no rework signal.
    assert measured_r_long(repo, fix, "calc.py", "add") == 0.0
