import pytest

from network_control_rl_framework.algebra import BaseNumber


def test_base_number_incorrect_init():
    # Base `q` is not prime or prime factor
    with pytest.raises(ValueError):
        BaseNumber(3, 14, 12)

    # Value `a` is too big for length `n`
    with pytest.raises(ValueError):
        BaseNumber(2, 2, 8)  # 1000 has size 4 bits, while `n` is 2


def test_base_number_operations():
    a = BaseNumber(3, 2, 5)  # 101
    b = BaseNumber(3, 2, 6)  # 110

    assert (a + b).a == 3  # 011
    assert (a - b).a == 3  # 011
    assert (b - a).a == 3  # 011


def test_base_number_operations_with_different_bases_and_sizes():
    # Different length `n`
    with pytest.raises(TypeError):
        a = BaseNumber(3, 2, 5)
        b = BaseNumber(2, 2, 1)

        a + b

    # Different bases q
    with pytest.raises(TypeError):
        a = BaseNumber(3, 2, 5)
        b = BaseNumber(3, 3, 1)

        a + b
