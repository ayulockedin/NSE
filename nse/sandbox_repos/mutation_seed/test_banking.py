from banking import (
    apply_fee,
    can_withdraw,
    deposit,
    is_overdrawn,
    tier,
    withdraw,
)


def test_can_withdraw():
    assert can_withdraw(100, 50) is True
    assert can_withdraw(100, 100) is True
    assert can_withdraw(100, 0) is False
    assert can_withdraw(100, 150) is False


def test_deposit():
    assert deposit(100, 50) == 150
    assert deposit(0, 0) == 0


def test_apply_fee():
    assert apply_fee(100, 10) == 90
    assert apply_fee(50, 0) == 50


def test_withdraw():
    assert withdraw(100, 30) == 70
    assert withdraw(100, 150) == 100  # rejected, balance unchanged
    assert withdraw(100, 0) == 100    # rejected, balance unchanged


def test_is_overdrawn():
    assert is_overdrawn(-1) is True
    assert is_overdrawn(0) is False
    assert is_overdrawn(10) is False


def test_tier():
    assert tier(5000) == 2
    assert tier(1000) == 2
    assert tier(500) == 1
    assert tier(100) == 1
    assert tier(50) == 0
