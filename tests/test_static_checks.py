from pathlib import Path

from nse.tools.static_checks import symbolic_check


def test_clean_repo_passes_gate(tmp_path: Path):
    (tmp_path / "ok.py").write_text("def f(x):\n    return x + 1\n")
    result = symbolic_check(tmp_path)
    assert result.syntax_ok


def test_syntax_error_fails_gate(tmp_path: Path):
    (tmp_path / "bad.py").write_text("def f(x):\n    return x +\n")
    result = symbolic_check(tmp_path)
    assert result.p_c == 0
    assert not result.syntax_ok
