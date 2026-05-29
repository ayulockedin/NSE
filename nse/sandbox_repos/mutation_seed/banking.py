"""Banking seed module for mutation dataset generation.

Boolean- and comparison-heavy (guards, tiers) with a call edge
(``withdraw`` -> ``can_withdraw``) so behaviour-changing mutants to the guard
ripple into the caller's tests.
"""

from __future__ import annotations


def can_withdraw(balance, amount):
    return amount > 0 and amount <= balance


def deposit(balance, amount):
    return balance + amount


def apply_fee(balance, fee):
    return balance - fee


def withdraw(balance, amount):
    if can_withdraw(balance, amount):
        return balance - amount
    return balance


def is_overdrawn(balance):
    return balance < 0


def tier(balance):
    if balance >= 1000:
        return 2
    if balance >= 100:
        return 1
    return 0
