import numpy as np
import pytest

import bag_of_characters as boc
from numpy.testing import assert_array_equal


def test_fit_1_gram():
    one_grammer = boc.WordNGramer(1)
    one_grammer.fit(["abacb"])
    assert one_grammer.corpus == ["a", "b", "c"]


def test_transform_of_1_grams():
    one_grammer = boc.WordNGramer(1)
    one_grammer.fit(["abacb"])
    X = one_grammer.transform(["aa", "a", "ababa", "cbc"])
    assert_array_equal(X[0], np.array([2, 0, 0]))
    assert_array_equal(X[1], np.array([1, 0, 0]))
    assert_array_equal(X[2], np.array([3, 2, 0]))
    assert_array_equal(X[3], np.array([0, 1, 2]))


def test_split_of_2_grams():
    two_gramer = boc.WordNGramer(2)
    assert list(two_gramer.split("a")) == []
    assert list(two_gramer.split("aba")) == ["ab", "ba"]
    assert list(two_gramer.split("abab")) == ["ab", "ba", "ab"]
    assert list(two_gramer.split("abacabe")) == ["ab", "ba", "ac", "ca", "ab", "be"]


def test_simple_fit_of_2_grams():
    two_gramer = boc.WordNGramer(2)
    two_gramer.fit(["aba"])
    assert two_gramer.corpus == ["ab", "ba"]


def test_fit_of_2_grams():
    two_gramer = boc.WordNGramer(2)
    two_gramer.fit(["aba", "ba", "a", "babab", "tralala"])
    assert two_gramer.corpus == ["ab", "al", "ba", "la", "ra", "tr"]


def test_transform_of_2_grams():
    two_gramer = boc.WordNGramer(2)
    two_gramer.fit(["aba", "ba", "a", "babab", "tralala"])
    X = two_gramer.transform(["trabra", "a", "ababa", "trabababara"])
    assert_array_equal(X[0], np.array([1, 0, 0, 0, 2, 1]))
    assert_array_equal(X[1], np.array([0, 0, 0, 0, 0, 0]))
    assert_array_equal(X[2], np.array([2, 0, 2, 0, 0, 0]))
    assert_array_equal(X[3], np.array([3, 0, 3, 0, 2, 1]))
