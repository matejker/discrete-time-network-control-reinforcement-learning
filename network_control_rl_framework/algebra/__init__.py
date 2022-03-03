from .primes import PRIMES, POWERS_2
from .finite_field import FiniteField as _FiniteField
from .binary_finite_field import BinaryFiniteField


class FiniteField:
    """Unifying  BinaryFiniteField and FiniteField under one class"""

    def __init__(self, a: int, q: int) -> None:
        if q in PRIMES:
            self.__class__ = _FiniteField  # type: ignore
        elif q in POWERS_2:
            self.__class__ = BinaryFiniteField  # type: ignore
        else:
            raise ValueError(f"{q} is neither of prime {PRIMES} or prime factor of 2 {POWERS_2}")
