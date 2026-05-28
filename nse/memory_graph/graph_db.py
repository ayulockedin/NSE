"""Memory Graph — NetworkX call/dependency graph over the repo (CPG-lite).

Nodes: functions, classes, modules. Edges: CALLS, IMPORTS, CONTAINS.
This is the MVP store; the query surface (``build_graph`` / ``get_subgraph`` /
``update_graph_on_patch``) is what the rest of the system depends on, so a
Neo4j backend can be swapped in later without changing call sites.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import networkx as nx

from nse.memory_graph.extractor import CodeNode, extract_file_nodes


class MemoryGraph:
    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root)
        self.g: nx.DiGraph = nx.DiGraph()
        # file -> tests that exercise it (populated by build_test_map).
        self.tests_map: dict[str, list[str]] = {}

    # ── build ──────────────────────────────────────────────────────────
    def build_graph(self) -> "MemoryGraph":
        name_to_id: dict[str, str] = {}
        all_nodes: list[CodeNode] = []
        for py in self.repo_root.rglob("*.py"):
            if any(p in py.parts for p in (".git", "venv", "__pycache__")):
                continue
            nodes = extract_file_nodes(py, self.repo_root)
            all_nodes.extend(nodes)
            for n in nodes:
                self.g.add_node(n.node_id, **n.__dict__)
                if n.kind == "function":
                    name_to_id.setdefault(n.name, n.node_id)

        # CONTAINS + CALLS + IMPORTS edges.
        for n in all_nodes:
            module_id = f"{n.file}::<module>"
            if n.node_id != module_id and self.g.has_node(module_id):
                self.g.add_edge(module_id, n.node_id, type="CONTAINS")
            for callee in n.calls:
                tgt = name_to_id.get(callee)
                if tgt:
                    self.g.add_edge(n.node_id, tgt, type="CALLS")
            for imp in n.imports:
                imp_module = f"{imp.replace('.', '/')}.py::<module>"
                if self.g.has_node(imp_module):
                    self.g.add_edge(n.node_id, imp_module, type="IMPORTS")
        return self

    def build_test_map(self) -> dict[str, list[str]]:
        """Map source files to the test node ids that collect against them.

        Uses ``pytest --collect-only -q`` as the blueprint suggests; falls back
        to a name-heuristic if pytest can't be invoked.
        """
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest", "--collect-only", "-q",
                    "--rootdir", str(self.repo_root), "-p", "no:cacheprovider", ".",
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            collected = [
                ln.strip()
                for ln in proc.stdout.splitlines()
                if "::" in ln
            ]
        except Exception:
            collected = []

        mapping: dict[str, list[str]] = {}
        for test_id in collected:
            test_file = test_id.split("::", 1)[0]
            # crude: a test_foo.py exercises foo.py
            stem = Path(test_file).stem.replace("test_", "")
            for py in self.repo_root.rglob(f"{stem}.py"):
                rel = py.relative_to(self.repo_root).as_posix()
                mapping.setdefault(rel, []).append(test_id)
        self.tests_map = mapping
        return mapping

    # ── query ──────────────────────────────────────────────────────────
    def get_subgraph(self, target_files: list[str], depth: int = 1) -> nx.DiGraph:
        seeds = [
            nid
            for nid, data in self.g.nodes(data=True)
            if data.get("file") in target_files
        ]
        keep: set[str] = set(seeds)
        frontier = set(seeds)
        for _ in range(depth):
            nxt: set[str] = set()
            for nid in frontier:
                nxt.update(self.g.successors(nid))
                nxt.update(self.g.predecessors(nid))
            keep |= nxt
            frontier = nxt
        return self.g.subgraph(keep).copy()

    def update_graph_on_patch(self, edited_files: list[str]) -> None:
        """Re-extract only the edited files and merge them back in."""
        for rel in edited_files:
            stale = [n for n, d in self.g.nodes(data=True) if d.get("file") == rel]
            self.g.remove_nodes_from(stale)
        # cheapest correct option for the MVP: rebuild.
        self.g.clear()
        self.build_graph()
