from strops import (
    count_vowels,
    is_palindrome,
    longest,
    repeat,
    reverse,
    starts_with,
)


def test_reverse():
    assert reverse("abc") == "cba"
    assert reverse("") == ""


def test_is_palindrome():
    assert is_palindrome("racecar") is True
    assert is_palindrome("abc") is False


def test_count_vowels():
    assert count_vowels("hello") == 2
    assert count_vowels("xyz") == 0
    assert count_vowels("aeiou") == 5


def test_repeat():
    assert repeat("ab", 3) == "ababab"
    assert repeat("x", 0) == ""


def test_starts_with():
    assert starts_with("hello", "he") is True
    assert starts_with("hello", "lo") is False


def test_longest():
    assert longest("abcd", "ab") == "abcd"
    assert longest("a", "bb") == "bb"
    assert longest("ab", "cd") == "ab"
