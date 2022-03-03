import pytest

from network_control_rl_framework.algebra import FiniteField


def test_incorrect_init_finite_field():
    with pytest.raises(ValueError):
        FiniteField(2, 4)

    with pytest.raises(ValueError):
        FiniteField(2, 60)


def test_finite_field_operations():
    a = FiniteField(2, 3)
    b = FiniteField(1, 3)

    assert (a + b).a == 0
    assert (a - b).a == 1
    assert (b - a).a == 2
    assert (a * b).a == 2
    assert (a / b).a == 2
    assert (~a).a == 2
    assert (a ** 3).a == 1
    