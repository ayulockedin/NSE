from listops import contains, count_positive, maximum, mean, minimum, total


def test_maximum():
    assert maximum([1, 5, 3]) == 5
    assert maximum([-2, -1, -3]) == -1


def test_minimum():
    assert minimum([1, 5, 3]) == 1
    assert minimum([-2, -1, -3]) == -3


def test_total():
    assert total([1, 2, 3]) == 6
    assert total([]) == 0


def test_count_positive():
    assert count_positive([1, -2, 3, 0]) == 2
    assert count_positive([-1, -2]) == 0


def test_mean():
    assert mean([2, 4, 6]) == 4
    assert mean([5]) == 5


def test_contains():
    assert contains([1, 2, 3], 2) is True
    assert contains([1, 2, 3], 9) is False
