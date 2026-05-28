from mathx import (
    add,
    all_positive,
    both,
    clamp,
    is_positive,
    mul,
    scale,
    sign,
    sub,
)


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_sub():
    assert sub(5, 2) == 3
    assert sub(0, 4) == -4


def test_mul():
    assert mul(3, 4) == 12
    assert mul(-2, 5) == -10


def test_is_positive():
    assert is_positive(1) is True
    assert is_positive(0) is False
    assert is_positive(-3) is False


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10


def test_sign():
    assert sign(7) == 1
    assert sign(-7) == -1
    assert sign(0) == 0


def test_all_positive():
    assert all_positive([1, 2, 3]) is True
    assert all_positive([1, -2, 3]) is False


def test_both():
    assert both(True, True) is True
    assert both(True, False) is False
    assert both(False, False) is False


def test_scale():
    assert scale(3) == 7
    assert scale(0) == 1
