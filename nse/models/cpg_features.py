"""CPG-lite graph featurizer shared by training and inference.

Both the dataset builder (offline labeling) and the orchestrator (online
scoring) call :func:`build_cpg_features` so the GNN sees the *same* node-feature
layout at train and serve time — no train/serve skew.

Given a :class:`~nse.memory_graph.graph_db.MemoryGraph` and the file(s) a patch
edits, we take a depth-1 subgraph and emit:

* ``node_features`` — ``(num_nodes, CPG_NODE_DIM)`` per-node feature rows, and
* ``edge_index``   — ``(2, num_edges)`` COO edges into that node ordering.

When the *edited function* is known we center the subgraph on that function node
(per-mutant granularity); otherwise we fall back to the whole-file subgraph.
Every feature is squashed into ``[0, 1]`` so the raw GNN input is bounded.
"""

from __future__ import annotations

import ast
from pathlib import Path

import networkx as nx

from nse.memory_graph.graph_db import MemoryGraph

# [is_function, is_class, is_module, complexity, loc, fan_out, fan_in, n_calls]
CPG_NODE_DIM = 8


def _empty_graph() -> tuple[list[list[float]], list[list[int]]]:
    """A single zero node with no edges — the neutral fallback when a file has
    no extractable structure (keeps tensor shapes valid downstream)."""
    return [[0.0] * CPG_NODE_DIM], [[], []]


def _featurize(sub: nx.DiGraph) -> tuple[list[list[float]], list[list[int]]]:
    nodes = list(sub.nodes(data=True))
    if not nodes:
        return _empty_graph()
    id_to_idx = {nid: i for i, (nid, _) in enumerate(nodes)}
    feats: list[list[float]] = []
    for nid, data in nodes:
        kind = data.get("kind", "")
        complexity = float(data.get("complexity", 1) or 1)
        loc = float(data.get("end_line", 1) - data.get("start_line", 1) + 1)
        n_calls = float(len(data.get("calls", []) or []))
        feats.append(
            [
                1.0 if kind == "function" else 0.0,
                1.0 if kind == "class" else 0.0,
                1.0 if kind == "module" else 0.0,
                min(1.0, complexity / 10.0),
                min(1.0, loc / 50.0),
                min(1.0, sub.out_degree(nid) / 10.0),
                min(1.0, sub.in_degree(nid) / 10.0),
                min(1.0, n_calls / 10.0),
            ]
        )
    edge_index: list[list[int]] = [[], []]
    for u, v in sub.edges():
        edge_index[0].append(id_to_idx[u])
        edge_index[1].append(id_to_idx[v])
    return feats, edge_index


def build_cpg_features(
    mg: MemoryGraph,
    target_files: list[str],
    center_functions: list[str] | None = None,
    depth: int = 1,
) -> tuple[list[list[float]], list[list[int]]]:
    """Return ``(node_features, edge_index)`` for the depth-1 subgraph.

    With ``center_functions`` the subgraph is seeded on those function nodes
    (``<file>::<name>``) for per-mutant granularity; without them, or if none
    resolve, it falls back to the whole-file subgraph. The graph reflects the
    code *before* a patch is applied, matching what the orchestrator has when it
    scores a branch.
    """
    if center_functions:
        seeds = [
            f"{rel}::{fn}"
            for rel in target_files
            for fn in center_functions
            if mg.g.has_node(f"{rel}::{fn}")
        ]
        if seeds:
            return _featurize(mg.subgraph_from_seeds(seeds, depth=depth))
    return _featurize(mg.get_subgraph(target_files, depth=depth))


def _calls_graph(mg: MemoryGraph) -> nx.DiGraph:
    """View of the graph restricted to CALLS edges (caller -> callee)."""
    call_edges = [
        (u, v) for u, v, d in mg.g.edges(data=True) if d.get("type") == "CALLS"
    ]
    return mg.g.edge_subgraph(call_edges) if call_edges else nx.DiGraph()


def blast_radius(mg: MemoryGraph, node_id: str, norm: float = 8.0) -> float:
    """Structural long-term-risk proxy in ``[0, 1]``: how many functions
    transitively *depend on* (call, directly or indirectly) ``node_id``. A leaf
    utility has a small blast radius; a heavily-depended-on core function a
    large one. Computed from CALLS ancestors only (CONTAINS/IMPORTS excluded)."""
    cg = _calls_graph(mg)
    if not cg.has_node(node_id):
        return 0.0
    return min(1.0, len(nx.ancestors(cg, node_id)) / norm)


def _function_sources(path: Path) -> dict[str, str]:
    """Map function name -> its exact source segment, for change detection."""
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(src, node)
            if seg is not None:
                out[node.name] = seg
    return out


def changed_function_names(
    pristine_dir: Path | str, patched_dir: Path | str, files: list[str]
) -> list[str]:
    """Names of functions whose source differs between the pristine and patched
    trees. Lets the orchestrator center the CPG subgraph on the *edited*
    function the same way the dataset builder does (train/serve consistency)."""
    pristine_dir, patched_dir = Path(pristine_dir), Path(patched_dir)
    changed: list[str] = []
    for rel in files:
        before = _function_sources(pristine_dir / rel)
        after = _function_sources(patched_dir / rel)
        for name, seg in after.items():
            if before.get(name) != seg:
                changed.append(name)
    return changed
