# Copyright cocotb contributors
# Licensed under the Revised BSD License, see LICENSE for details.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from cocotb.types import Array, Range


def test_value_only_construction():
    a = Array("1234")
    assert a.left == 0
    assert a.direction == "to"
    assert a.right == 3


def test_both_construction():
    a = Array("1234", Range(-2, 1))
    assert a.left == -2
    assert a.direction == "to"
    assert a.right == 1

    Array("1234", 4)
    Array("1234", range=Range(-2, 1))


def test_range_int_construction():
    assert Array("1234", 4)


def test_bad_construction():
    with pytest.raises(TypeError):
        Array()
    with pytest.raises(TypeError):
        Array(value=1)
    with pytest.raises(TypeError):
        Array(range=Range(0, 1))
    with pytest.raises(ValueError):
        Array(value="1234", range=Range(0, 1))
    with pytest.raises(TypeError):
        Array(value="1234", range=object())
    with pytest.raises(TypeError):
        Array("1234", range(4))


def test_length():
    a = Array("123456")
    assert len(a) == 6


def test_range():
    r = Range(-2, 1)
    a = Array(value="0123", range=r)
    assert a.range == r


def test_equality():
    assert Array("1234", Range(1, 4)) == Array("1234", Range(1, 4))
    assert Array("1234", Range(1, 4)) == Array("1234", Range(0, -3))
    assert Array("1234", Range(1, 4)) != Array("4321", Range(1, 4))
    assert Array("1234") != Array("12")
    assert Array("1234") != 8
    assert Array([1, 2, 3, 4]) == [1, 2, 3, 4]


def test_repr_eval():
    r = Array("1234")
    assert eval(repr(r)) == r


def test_iter():
    val = [7, True, object(), "example"]
    a = Array(val)
    assert list(a) == val


def test_reversed():
    val = [7, True, object(), "example"]
    a = Array(val)
    assert list(reversed(a)) == list(reversed(val))


def test_contains():
    a = Array([7, True, object(), "example"])
    assert True in a
    assert 89 not in a


def test_index():
    r = Array(i for j in range(10) for i in range(10))  # 0-9 repeated 10 times
    assert r.index(5) == 5
    assert r.index(5, 10, 20) == 15
    with pytest.raises(IndexError):
        r.index(object())


def test_count():
    assert Array("011X1Z").count("1") == 3


def test_indexing():
    a = Array("1234", Range(8, "to", 11))
    assert a[8] == "1"
    with pytest.raises(IndexError):
        a[0]
    a[11] = False
    assert a[11] is False

    b = Array("1234", Range(10, "downto", 7))
    assert b[8] == "3"
    with pytest.raises(IndexError):
        b[-2]
    b[8] = 9
    assert b[8] == 9


def test_bad_indexing():
    with pytest.raises(TypeError):
        Array("1234")[list()]
    with pytest.raises(TypeError):
        Array("1234")[object()] = 9


def test_slicing():
    a = Array("testingstuff")
    b = a[2:6]
    assert b.left == 2
    assert b.right == 6
    assert b == Array("sting")
    a[0:3] = "hack"
    assert a == Array("hackingstuff")


def test_slicing_infered_start_stop():
    a = Array([1, 2, 3, 4])
    assert a[:] == a
    a[:] = "1234"
    assert a == Array("1234")


def test_dont_specify_step():
    with pytest.raises(IndexError):
        Array("1234")[::1]
    with pytest.raises(IndexError):
        Array("7896")[1:2:1] = [1, 2]


def test_slice_direction_mismatch():
    a = Array([1, 2, 3, 4], Range(10, "downto", 7))
    with pytest.raises(IndexError):
        a[7:9]
    with pytest.raises(IndexError):
        a[9:10] = ["a", "b"]


def test_set_slice_wrong_length():
    a = Array("example")
    with pytest.raises(ValueError):
        a[2:4] = "real bad"


def test_slice_correct_infered():
    a = Array("1234")
    b = a[:0]
    assert b.right == 0


def test_changing_range():
    a = Array("1234")
    a.range = Range(3, "downto", 0)
    assert a.range == Range(3, "downto", 0)
    with pytest.raises(TypeError):
        a.range = range(10)
    with pytest.raises(ValueError):
        a.range = Range(7, "downto", 0)
