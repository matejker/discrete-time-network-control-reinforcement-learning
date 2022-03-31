import pytest
import numpy as np

from network_control_rl.algebra import BaseNumber, BaseNumberTypeError, BaseNumberValueError


def test_base_number_incorrect_init():
    # Base `q` is not prime or prime factor
    with pytest.raises(BaseNumberValueError):
        BaseNumber(3, 14, 12)

    # Value `a` is too big for length `n`
    with pytest.raises(BaseNumberValueError):
        BaseNumber(2, 2, 8)  # 1000 has size 4 bits, while `n` is 2


def test_base_number_operations():
    a = BaseNumber(3, 2, 5)  # 101
    b = BaseNumber(3, 2, 6)  # 110

    assert (a + b) == 3  # 011
    assert (a - b) == 3  # 011
    assert (b - a) == 3  # 011


def test_base_number_operations_with_different_bases_and_sizes():
    # Different length `n`
    with pytest.raises(BaseNumberTypeError):
        a = BaseNumber(3, 2, 5)
        b = BaseNumber(2, 2, 1)

        a + b

    # Different bases q
    with pytest.raises(BaseNumberTypeError):
        a = BaseNumber(3, 2, 5)
        b = BaseNumber(3, 3, 1)

        a + b


def test_base_number_array_convert():
    a = BaseNumber(3, 2, 5)
    assert (a.to_array() == np.array([1, 0, 1], dtype=np.int8)).all()

    a.from_array(np.array([1, 0, 0, 0], dtype=np.int8))
    assert a.a == 8
    assert a.n == 4
    assert a.q == 2


def test_base_number_to_string():
    a = BaseNumber(3, 16, 33)
    assert a.to_string() == "021"
    assert a.__repr__() == "BaseNumber(a=021_16=33_10, n=3, q=16)"
    assert str(a) == "33"
