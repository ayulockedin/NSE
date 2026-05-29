from geometry import (
    is_square,
    rect_area,
    rect_perimeter,
    scaled_square,
    square_area,
    triangle_area,
)


def test_square_area():
    assert square_area(3) == 9
    assert square_area(0) == 0


def test_rect_area():
    assert rect_area(2, 5) == 10
    assert rect_area(4, 0) == 0


def test_rect_perimeter():
    assert rect_perimeter(2, 3) == 10
    assert rect_perimeter(0, 0) == 0


def test_triangle_area():
    assert triangle_area(4, 3) == 6
    assert triangle_area(0, 5) == 0


def test_is_square():
    assert is_square(4, 4) is True
    assert is_square(4, 5) is False


def test_scaled_square():
    assert scaled_square(3, 2) == 18
    assert scaled_square(2, 0) == 0
