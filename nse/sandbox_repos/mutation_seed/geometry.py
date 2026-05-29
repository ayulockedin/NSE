"""Geometry seed module for mutation dataset generation.

Arithmetic-heavy with an intra-module call edge (``scaled_square`` ->
``square_area``) so the CPG-lite graph for this file has real CALLS structure,
not just isolated nodes.
"""

from __future__ import annotations


def square_area(side):
    return side * side


def rect_area(w, h):
    return w * h


def rect_perimeter(w, h):
    return 2 * (w + h)


def triangle_area(base, height):
    return base * height / 2


def is_square(w, h):
    return w == h


def scaled_square(side, k):
    return square_area(side) * k
