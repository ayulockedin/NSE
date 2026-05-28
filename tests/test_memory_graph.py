from pathlib import Path

from nse.memory_graph.context import get_prompt_context
from nse.memory_graph.graph_db import MemoryGraph


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(
        "def helper(x):\n    return x * 2\n\n"
        "def main(y):\n    return helper(y) + 1\n"
    )
    return tmp_path


def test_build_graph_finds_functions(tmp_path: Path):
    mg = MemoryGraph(_make_repo(tmp_path)).build_graph()
    fns = [d["name"] for _, d in mg.g.nodes(data=True) if d.get("kind") == "function"]
    assert "helper" in fns and "main" in fns


def test_call_edge_present(tmp_path: Path):
    mg = MemoryGraph(_make_repo(tmp_path)).build_graph()
    call_edges = [
        (u, v) for u, v, d in mg.g.edges(data=True) if d.get("type") == "CALLS"
    ]
    assert any(u.endswith("::main") and v.endswith("::helper") for u, v in call_edges)


def test_context_assembly(tmp_path: Path):
    mg = MemoryGraph(_make_repo(tmp_path)).build_graph()
    ctx = get_prompt_context(mg, ["a.py"], depth=1)
    assert ctx["target_files"] == ["a.py"]
    assert any(n["kind"] == "function" for n in ctx["nodes"])
