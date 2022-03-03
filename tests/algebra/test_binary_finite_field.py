import pytest

from network_control_rl_framework.algebra import BinaryFiniteField


def test_incorrect_power_of_2():
    # 128 is too big
    with pytest.raises(ValueError):
        BinaryFiniteField(1, 128)

    # 7 is not power of 2
    with pytest.raises(ValueError):
        BinaryFiniteField(1, 7)

    # 12 is not "prime factor"
    with pytest.raises(ValueError):
        BinaryFiniteField(1, 12)


def test_operations():
    a = BinaryFiniteField(2, 2)  # x
    b = BinaryFiniteField(3, 2)  # x+1

    assert (a + b).a == 1
    assert (a - b).a == 2
    assert (b - a).a == 1
    assert (a * b).a == 1
    assert (a / b).a == 3  # x + 1
    assert (~b).a == 2  # x
