"""Richer seed module for mutation-based dataset generation.

Deliberately uses a mix of arithmetic, comparison, boolean, and constant
expressions so the mutation engine has many distinct sites to perturb. The
accompanying tests are thorough enough that *behaviour-changing* mutants are
reliably killed (label 0) while benign edits survive (label 1).
"""

from __future__ import annotations


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def is_positive(x):
    return x > 0


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def all_positive(xs):
    return all(v > 0 for v in xs)


def both(a, b):
    return a and b


def scale(x):
    return x * 2 + 1
