"""Tests for the AST mutation engine."""

import ast

from nse.data.mutate import generate_mutations

SAMPLE = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "def gt(a, b):\n"
    "    return a > b\n"
    "\n"
    "def both(a, b):\n"
    "    return a and b\n"
    "\n"
    "def scale(x):\n"
    "    return x * 2\n"
)


def test_generates_mutants():
    muts = generate_mutations(SAMPLE)
    assert len(muts) > 0


def test_all_mutants_parse_and_differ():
    baseline = ast.unparse(ast.parse(SAMPLE))
    for mut in generate_mutations(SAMPLE):
        # Every mutant must be valid Python ...
        ast.parse(mut.mutated_src)
        # ... and actually different from the original.
        assert mut.mutated_src != baseline


def test_covers_each_operator_family():
    kinds = {m.kind for m in generate_mutations(SAMPLE)}
    assert {"binop", "compare", "boolop", "constant", "benign_noop"} <= kinds


def test_benign_mutants_are_marked_non_breaking():
    benign = [m for m in generate_mutations(SAMPLE) if m.kind == "benign_noop"]
    assert benign
    assert all(not m.assumed_breaking for m in benign)
    # Each benign mutant inserts the no-op binding.
    assert all("_nse_noop" in m.mutated_src for m in benign)


def test_breaking_mutants_are_marked_breaking():
    breaking = [m for m in generate_mutations(SAMPLE) if m.kind != "benign_noop"]
    assert breaking
    assert all(m.assumed_breaking for m in breaking)


def test_mutants_are_deduplicated():
    muts = generate_mutations(SAMPLE)
    srcs = [m.mutated_src for m in muts]
    assert len(srcs) == len(set(srcs))


def test_mutants_record_enclosing_function():
    muts = generate_mutations(SAMPLE)
    # Every site in SAMPLE lives inside a function, so none should be "".
    assert all(m.function for m in muts)
    # The functions covered are exactly SAMPLE's definitions.
    assert {m.function for m in muts} <= {"add", "gt", "both", "scale"}
    # A binop mutant of `a + b` must be attributed to `add`.
    add_binops = [m for m in muts if m.kind == "binop" and m.function == "add"]
    assert add_binops
