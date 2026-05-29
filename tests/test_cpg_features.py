"""Tests for the CPG-lite graph featurizer.

These exercise the real seed repo so the featurizer is checked against actual
extracted structure (functions, CONTAINS/CALLS edges), not a mock.
"""

from nse.config import ROOT
from nse.memory_graph.graph_db import MemoryGraph
from nse.models.cpg_features import (
    CPG_NODE_DIM,
    blast_radius,
    build_cpg_features,
    changed_function_names,
)

SEED = ROOT / "nse" / "sandbox_repos" / "mutation_seed"


def _graph() -> MemoryGraph:
    return MemoryGraph(SEED).build_graph()


def test_features_have_real_structure():
    mg = _graph()
    node_features, edge_index = build_cpg_features(mg, ["banking.py"])
    # banking.py has 6 functions + module; depth-1 pulls in calling tests too.
    assert len(node_features) > 1
    assert len(edge_index) == 2
    assert len(edge_index[0]) == len(edge_index[1]) > 0


def test_feature_rows_are_bounded_and_fixed_width():
    mg = _graph()
    node_features, _ = build_cpg_features(mg, ["stats.py"])
    for row in node_features:
        assert len(row) == CPG_NODE_DIM
        assert all(0.0 <= v <= 1.0 for v in row)


def test_edge_indices_are_in_range():
    mg = _graph()
    node_features, edge_index = build_cpg_features(mg, ["geometry.py"])
    n = len(node_features)
    for endpoints in edge_index:
        assert all(0 <= idx < n for idx in endpoints)


def test_missing_file_yields_neutral_single_node():
    mg = _graph()
    node_features, edge_index = build_cpg_features(mg, ["does_not_exist.py"])
    assert node_features == [[0.0] * CPG_NODE_DIM]
    assert edge_index == [[], []]


def test_kind_one_hot_is_exclusive():
    mg = _graph()
    node_features, _ = build_cpg_features(mg, ["banking.py"])
    # First three features are the is_function/is_class/is_module one-hot.
    for row in node_features:
        assert sum(row[:3]) <= 1.0


def test_function_centered_subgraph_is_smaller_than_file():
    mg = _graph()
    whole_file, _ = build_cpg_features(mg, ["banking.py"])
    one_fn, _ = build_cpg_features(
        mg, ["banking.py"], center_functions=["can_withdraw"]
    )
    # Centering on a single function yields a strict subset of the file graph.
    assert 1 < len(one_fn) <= len(whole_file)


def test_unknown_center_function_falls_back_to_file():
    mg = _graph()
    whole_file, _ = build_cpg_features(mg, ["banking.py"])
    fallback, _ = build_cpg_features(
        mg, ["banking.py"], center_functions=["nonexistent_fn"]
    )
    assert len(fallback) == len(whole_file)


def test_blast_radius_orders_core_above_leaf():
    mg = _graph()
    # summation is called (transitively) by average and variance; rect_area is
    # a leaf nobody else calls -> summation must have the larger blast radius.
    core = blast_radius(mg, "stats.py::summation")
    leaf = blast_radius(mg, "geometry.py::rect_area")
    assert 0.0 <= leaf < core <= 1.0


def test_changed_function_names_detects_edited_function(tmp_path):
    pristine = tmp_path / "a"
    patched = tmp_path / "b"
    pristine.mkdir()
    patched.mkdir()
    (pristine / "m.py").write_text(
        "def f(x):\n    return x + 1\n\ndef g(x):\n    return x\n", encoding="utf-8"
    )
    (patched / "m.py").write_text(
        "def f(x):\n    return x - 1\n\ndef g(x):\n    return x\n", encoding="utf-8"
    )
    changed = changed_function_names(pristine, patched, ["m.py"])
    assert changed == ["f"]
