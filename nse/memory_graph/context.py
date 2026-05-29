"""Context assembly — turn a subgraph into a compact prompt payload.

``get_prompt_context`` returns signatures, docstring snippets, affected tests,
and dependency edges — never full function bodies by default. If the serialised
payload would blow the token budget, the least-relevant nodes (lowest degree)
are pruned first.
"""

from __future__ import annotations

from typing import Any

from nse.config import SETTINGS
from nse.memory_graph.graph_db import MemoryGraph


def _approx_tokens(text: str) -> int:
    # ~4 chars/token heuristic; good enough for budget pruning.
    return max(1, len(text) // 4)


def get_prompt_context(
    mg: MemoryGraph,
    target_files: list[str],
    depth: int = 1,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    max_tokens = max_tokens or SETTINGS.budgets.max_total_llm_tokens_per_task
    sub = mg.get_subgraph(target_files, depth=depth)

    nodes: list[dict[str, Any]] = []
    for nid, data in sub.nodes(data=True):
        if data.get("kind") == "module":
            continue
        nodes.append(
            {
                "id": nid,
                "kind": data.get("kind"),
                "signature": data.get("signature") or data.get("name"),
                "doc": data.get("docstring", "")[:120],
                "complexity": data.get("complexity", 1),
                "degree": sub.degree(nid),
            }
        )

    # Prune least-relevant (lowest degree) until under budget.
    nodes.sort(key=lambda n: n["degree"], reverse=True)
    edges = [
        {"src": u, "dst": v, "type": d.get("type")}
        for u, v, d in sub.edges(data=True)
    ]
    affected_tests = sorted(
        {t for f in target_files for t in mg.tests_map.get(f, [])}
    )

    def serialised(ns: list[dict[str, Any]]) -> str:
        return repr(ns) + repr(edges) + repr(affected_tests)

    while nodes and _approx_tokens(serialised(nodes)) > max_tokens:
        nodes.pop()  # drop lowest-degree node

    # Current source of the edited files, so a real model can produce a faithful
    # full_file_rewrite instead of guessing (capped to bound context).
    source: dict[str, str] = {}
    for f in target_files:
        path = mg.repo_root / f
        if path.exists():
            source[f] = path.read_text(encoding="utf-8")[:6000]

    return {
        "target_files": target_files,
        "nodes": nodes,
        "edges": edges,
        "affected_tests": affected_tests,
        "source": source,
    }
