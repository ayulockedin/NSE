"""String/seed module for mutation dataset generation."""

from __future__ import annotations


def reverse(s):
    return s[::-1]


def is_palindrome(s):
    return s == s[::-1]


def count_vowels(s):
    n = 0
    for ch in s:
        if ch in "aeiou":
            n = n + 1
    return n


def repeat(s, k):
    return s * k


def starts_with(s, prefix):
    return s[: len(prefix)] == prefix


def longest(a, b):
    if len(a) >= len(b):
        return a
    return b
