"""List/numeric seed module for mutation dataset generation."""

from __future__ import annotations


def maximum(xs):
    best = xs[0]
    for v in xs:
        if v > best:
            best = v
    return best


def minimum(xs):
    best = xs[0]
    for v in xs:
        if v < best:
            best = v
    return best


def total(xs):
    acc = 0
    for v in xs:
        acc = acc + v
    return acc


def count_positive(xs):
    n = 0
    for v in xs:
        if v > 0:
            n = n + 1
    return n


def mean(xs):
    return total(xs) / len(xs)


def contains(xs, target):
    for v in xs:
        if v == target:
            return True
    return False
