"""AST-based mutation engine.

Given a known-good Python source string, produce a set of *mutants* — each a
single, targeted edit. Two families:

* **breaking** edits (operator swaps, constant bumps, boolean flips) that change
  behaviour and should usually be killed by a good test suite, and
* **benign** edits (no-op statement insertion) that preserve behaviour.

``assumed_breaking`` is only a hint for reporting/balance. The *true* label of a
mutant comes from running it through the sandbox (see ``nse.data.dataset``) —
mutation testing's whole point is that some "breaking" edits survive weak tests.

We mutate one site per mutant by walking the tree deterministically (``ast.walk``
is BFS and stable), deep-copying, and editing the node at a fixed index. This
keeps the original-vs-copy indexing aligned without mutating shared state.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Iterator
from dataclasses import dataclass

# Operator swap tables. Each maps an AST op type -> its replacement type.
_BINOP_SWAP: dict[type, type] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
}
_CMP_SWAP: dict[type, type] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.Gt: ast.LtE,
    ast.LtE: ast.Gt,
}
_BOOL_SWAP: dict[type, type] = {
    ast.And: ast.Or,
    ast.Or: ast.And,
}


@dataclass(frozen=True)
class Mutation:
    """A single mutant of a source file."""

    kind: str            # "binop" | "compare" | "constant" | "boolop" | "benign_noop"
    description: str     # human-readable, e.g. "Add->Sub @ binop#0"
    mutated_src: str     # full mutated source
    assumed_breaking: bool
    function: str = ""   # enclosing function name ("" = module scope)


def _scope_map(tree: ast.AST) -> dict[int, str]:
    """Map ``id(node) -> enclosing function name`` for every node in ``tree``.

    Walks top-down so an inner function overrides the outer one, giving the
    *innermost* enclosing function (module-level nodes map to "")."""
    out: dict[int, str] = {}

    def visit(node: ast.AST, current: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in ast.walk(child):
                    out[id(d)] = child.name
                visit(child, child.name)
            else:
                visit(child, current)

    visit(tree, "")
    return out


def _enclosing_function(tree: ast.AST, node: ast.AST) -> str:
    return _scope_map(tree).get(id(node), "")


def _unparse(tree: ast.AST) -> str:
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def normalized_source(source: str) -> str:
    """Round-trip source through the AST so it matches the formatting of every
    mutant. Diffing a mutant against this baseline (rather than the original
    hand-written source) isolates the mutation instead of drowning it in
    whole-file reformatting noise from ``ast.unparse``."""
    return _unparse(ast.parse(source))


def _nodes(tree: ast.AST, predicate) -> list[ast.AST]:
    return [n for n in ast.walk(tree) if predicate(n)]


def _binop_mutants(tree: ast.Module) -> Iterator[Mutation]:
    def pred(n: ast.AST) -> bool:
        return isinstance(n, ast.BinOp) and type(n.op) in _BINOP_SWAP

    for i in range(len(_nodes(tree, pred))):
        new = copy.deepcopy(tree)
        node = _nodes(new, pred)[i]
        fn = _enclosing_function(new, node)
        old_name = type(node.op).__name__
        node.op = _BINOP_SWAP[type(node.op)]()  # type: ignore[union-attr]
        yield Mutation(
            kind="binop",
            description=f"{old_name}->{type(node.op).__name__} @ binop#{i}",
            mutated_src=_unparse(new),
            assumed_breaking=True,
            function=fn,
        )


def _compare_mutants(tree: ast.Module) -> Iterator[Mutation]:
    # A Compare node may chain several ops; mutate the first swappable op.
    def pred(n: ast.AST) -> bool:
        return isinstance(n, ast.Compare) and any(
            type(op) in _CMP_SWAP for op in n.ops
        )

    for i in range(len(_nodes(tree, pred))):
        new = copy.deepcopy(tree)
        node = _nodes(new, pred)[i]
        fn = _enclosing_function(new, node)
        for j, op in enumerate(node.ops):  # type: ignore[union-attr]
            if type(op) in _CMP_SWAP:
                old_name = type(op).__name__
                node.ops[j] = _CMP_SWAP[type(op)]()  # type: ignore[union-attr]
                yield Mutation(
                    kind="compare",
                    description=f"{old_name}->{type(node.ops[j]).__name__} @ cmp#{i}",
                    mutated_src=_unparse(new),
                    assumed_breaking=True,
                    function=fn,
                )
                break


def _boolop_mutants(tree: ast.Module) -> Iterator[Mutation]:
    def pred(n: ast.AST) -> bool:
        return isinstance(n, ast.BoolOp) and type(n.op) in _BOOL_SWAP

    for i in range(len(_nodes(tree, pred))):
        new = copy.deepcopy(tree)
        node = _nodes(new, pred)[i]
        fn = _enclosing_function(new, node)
        old_name = type(node.op).__name__
        node.op = _BOOL_SWAP[type(node.op)]()  # type: ignore[union-attr]
        yield Mutation(
            kind="boolop",
            description=f"{old_name}->{type(node.op).__name__} @ boolop#{i}",
            mutated_src=_unparse(new),
            assumed_breaking=True,
            function=fn,
        )


def _is_number(n: ast.AST) -> bool:
    return (
        isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
    )


def _constant_mutants(tree: ast.Module) -> Iterator[Mutation]:
    for i in range(len(_nodes(tree, _is_number))):
        new = copy.deepcopy(tree)
        node = _nodes(new, _is_number)[i]
        fn = _enclosing_function(new, node)
        old = node.value  # type: ignore[union-attr]
        node.value = old + 1  # type: ignore[union-attr]
        yield Mutation(
            kind="constant",
            description=f"{old}->{old + 1} @ const#{i}",
            mutated_src=_unparse(new),
            assumed_breaking=True,
            function=fn,
        )

    def is_bool(n: ast.AST) -> bool:
        return isinstance(n, ast.Constant) and isinstance(n.value, bool)

    for i in range(len(_nodes(tree, is_bool))):
        new = copy.deepcopy(tree)
        node = _nodes(new, is_bool)[i]
        fn = _enclosing_function(new, node)
        old = node.value  # type: ignore[union-attr]
        node.value = not old  # type: ignore[union-attr]
        yield Mutation(
            kind="constant",
            description=f"{old}->{not old} @ bool#{i}",
            mutated_src=_unparse(new),
            assumed_breaking=True,
            function=fn,
        )


def _benign_mutants(tree: ast.Module) -> Iterator[Mutation]:
    """Insert a no-op binding at the top of each function body.

    Behaviour-preserving, so these are expected to keep the tests green and
    supply the positive class. They still produce a non-trivial diff, so the
    patch-feature vector is meaningful.
    """

    def pred(n: ast.AST) -> bool:
        return isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))

    for i in range(len(_nodes(tree, pred))):
        new = copy.deepcopy(tree)
        fn = _nodes(new, pred)[i]
        noop = ast.parse("_nse_noop = 0").body[0]
        fn.body.insert(0, noop)  # type: ignore[union-attr]
        yield Mutation(
            kind="benign_noop",
            description=f"noop in {fn.name}",  # type: ignore[union-attr]
            mutated_src=_unparse(new),
            assumed_breaking=False,
            function=fn.name,  # type: ignore[union-attr]
        )


def generate_mutations(source: str) -> list[Mutation]:
    """Return all mutants of ``source``. Mutants identical to the input or to
    each other are de-duplicated (an operator swap can be a syntactic no-op)."""
    tree = ast.parse(source)
    baseline = _unparse(copy.deepcopy(tree))

    seen: set[str] = {baseline}
    out: list[Mutation] = []
    for gen in (
        _binop_mutants,
        _compare_mutants,
        _boolop_mutants,
        _constant_mutants,
        _benign_mutants,
    ):
        for mut in gen(tree):
            if mut.mutated_src in seen:
                continue
            seen.add(mut.mutated_src)
            out.append(mut)
    return out
