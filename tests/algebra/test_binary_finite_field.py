import pytest

from network_control_rl.algebra import BinaryFiniteField


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
    a = BinaryFiniteField(2, 4)  # x
    b = BinaryFiniteField(3, 4)  # x+1

    assert a + b == 1
    assert a - b == 2
    assert b - a == 1
    assert a * b == 1
    assert a / b == 3  # x + 1
    assert ~b == 2  # x
