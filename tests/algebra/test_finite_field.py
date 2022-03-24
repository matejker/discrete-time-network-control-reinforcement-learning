import pytest

from network_control_rl_framework.algebra import FiniteField, FiniteFieldValueError
from network_control_rl_framework.algebra.finite_field import FiniteField as ff
from network_control_rl_framework.algebra.binary_finite_field import BinaryFiniteField as bff


def test_incorrect_init_finite_field():
    with pytest.raises(FiniteFieldValueError):
        ff(2, 4)

    with pytest.raises(FiniteFieldValueError):
        ff(2, 60)


def test_finite_field_operations():
    a = ff(2, 3)
    b = ff(1, 3)

    assert (a + b).a == 0
    assert (a - b).a == 1
    assert (b - a).a == 2
    assert (a * b).a == 2
    assert (a / b).a == 2
    assert (~a).a == 2
    assert (a**3).a == 1


def test_finite_field_split():
    prime = FiniteField(2, 3)
    prime_factor = FiniteField(2, 8)

    assert prime.__class__ == ff
    assert prime_factor.__class__ == bff
