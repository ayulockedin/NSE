from stats import average, count_above, in_range, summation, variance


def test_summation():
    assert summation([1, 2, 3]) == 6
    assert summation([]) == 0


def test_average():
    assert average([2, 4, 6]) == 4
    assert average([5]) == 5


def test_variance():
    assert variance([2, 2, 2]) == 0
    assert variance([1, 3]) == 1


def test_in_range():
    assert in_range(5, 0, 10) is True
    assert in_range(0, 0, 10) is True
    assert in_range(10, 0, 10) is True
    assert in_range(-1, 0, 10) is False
    assert in_range(11, 0, 10) is False


def test_count_above():
    assert count_above([1, 2, 3, 4], 2) == 2
    assert count_above([1, 1, 1], 5) == 0
