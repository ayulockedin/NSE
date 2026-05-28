"""AST extraction via tree-sitter (with a stdlib ``ast`` fallback).

Extracts the node-level facts the Memory Graph needs: functions, classes,
imports, and intra-file call edges. Node IDs are stable: ``<relpath>::<qualname>``
keyed by file path + symbol so they survive across edits.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

# tree-sitter is the blueprint's primary parser; we degrade to stdlib ast for
# Python so the extractor works even before grammars are built.
try:
    import tree_sitter_python as tspython  # type: ignore
    from tree_sitter import Language, Parser  # type: ignore

    _TS_LANG = Language(tspython.language())
    _TS_PARSER = Parser(_TS_LANG)
    _TS_AVAILABLE = True
except Exception:  # pragma: no cover - any import/build failure
    _TS_AVAILABLE = False


@dataclass
class CodeNode:
    node_id: str
    kind: str              # "function" | "class" | "module"
    name: str
    file: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    complexity: int = 1    # cyclomatic-ish
    calls: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


def _qualname(stack: list[str], name: str) -> str:
    return ".".join([*stack, name]) if stack else name


def _cyclomatic(node: ast.AST) -> int:
    branch_types = (
        ast.If, ast.For, ast.While, ast.And, ast.Or,
        ast.ExceptHandler, ast.With, ast.comprehension, ast.BoolOp,
    )
    return 1 + sum(isinstance(n, branch_types) for n in ast.walk(node))


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in fn.args.args]
    ret = ast.unparse(fn.returns) if fn.returns else ""
    sig = f"{fn.name}({', '.join(args)})"
    return f"{sig} -> {ret}" if ret else sig


def _calls_in(node: ast.AST) -> list[str]:
    out: list[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.append(f.id)
            elif isinstance(f, ast.Attribute):
                out.append(f.attr)
    return out


def extract_file_nodes(filepath: Path | str, repo_root: Path | str) -> list[CodeNode]:
    """Return the :class:`CodeNode` list for one Python file.

    Uses stdlib ``ast`` for structure (robust + zero-build); tree-sitter
    availability is recorded for callers that want raw CST access.
    """
    filepath = Path(filepath)
    repo_root = Path(repo_root)
    rel = filepath.relative_to(repo_root).as_posix()
    src = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(filepath))
    except SyntaxError:
        return []

    nodes: list[CodeNode] = []
    module_imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            module_imports.append(node.module or "")

    nodes.append(
        CodeNode(
            node_id=f"{rel}::<module>",
            kind="module",
            name=rel,
            file=rel,
            start_line=1,
            end_line=len(src.splitlines()) or 1,
            imports=module_imports,
        )
    )

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = _qualname(stack, child.name)
                nodes.append(
                    CodeNode(
                        node_id=f"{rel}::{qn}",
                        kind="function",
                        name=child.name,
                        file=rel,
                        start_line=child.lineno,
                        end_line=getattr(child, "end_lineno", child.lineno),
                        signature=_signature(child),
                        docstring=(ast.get_docstring(child) or "")[:200],
                        complexity=_cyclomatic(child),
                        calls=sorted(set(_calls_in(child))),
                    )
                )
                visit(child, [*stack, child.name])
            elif isinstance(child, ast.ClassDef):
                qn = _qualname(stack, child.name)
                nodes.append(
                    CodeNode(
                        node_id=f"{rel}::{qn}",
                        kind="class",
                        name=child.name,
                        file=rel,
                        start_line=child.lineno,
                        end_line=getattr(child, "end_lineno", child.lineno),
                        docstring=(ast.get_docstring(child) or "")[:200],
                    )
                )
                visit(child, [*stack, child.name])
            else:
                visit(child, stack)

    visit(tree, [])
    return nodes


def tree_sitter_available() -> bool:
    return _TS_AVAILABLE
