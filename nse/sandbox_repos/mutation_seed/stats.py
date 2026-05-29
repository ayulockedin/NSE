"""Statistics seed module for mutation dataset generation.

Has a small call chain (``variance`` -> ``average`` -> ``summation``) giving the
CPG-lite graph a multi-hop CALLS path, plus boolean/compare sites in
``in_range`` and ``count_above``.
"""

from __future__ import annotations


def summation(xs):
    acc = 0
    for v in xs:
        acc = acc + v
    return acc


def average(xs):
    return summation(xs) / len(xs)


def variance(xs):
    m = average(xs)
    acc = 0
    for v in xs:
        acc = acc + (v - m) * (v - m)
    return acc / len(xs)


def in_range(x, lo, hi):
    return x >= lo and x <= hi


def count_above(xs, threshold):
    n = 0
    for v in xs:
        if v > threshold:
            n = n + 1
    return n
