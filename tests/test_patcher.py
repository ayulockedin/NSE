import shutil
from pathlib import Path

import pytest

from nse.orchestrator.patcher import (
    PatchSafetyError,
    apply_patch_strict_then_fallback,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    return tmp_path


def test_tier1_git_apply(repo: Path):
    patch = (
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,3 @@\n"
        " def add(a, b):\n+    # checked\n     return a + b\n"
    )
    result = apply_patch_strict_then_fallback(repo, patch_text=patch)
    assert result["mode"] in {"git_apply", "git_3way_apply", "fuzzy_apply"}
    assert "# checked" in (repo / "calc.py").read_text()


def test_full_rewrite_tier3(repo: Path):
    result = apply_patch_strict_then_fallback(
        repo, full_rewrites={"calc.py": "def add(a, b):\n    return a + b + 0\n"}
    )
    assert result["mode"] == "full_rewrite"


def test_sensitive_path_blocked(repo: Path):
    with pytest.raises(PatchSafetyError):
        apply_patch_strict_then_fallback(
            repo, full_rewrites={".env": "SECRET=1\n"}
        )


def test_oversized_rewrite_blocked(repo: Path):
    big = "x = 1\n" * 400
    with pytest.raises(PatchSafetyError):
        apply_patch_strict_then_fallback(repo, full_rewrites={"big.py": big})
