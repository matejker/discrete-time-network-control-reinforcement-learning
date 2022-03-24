from math import log2

from .primes import PRIMES, POWERS_2
from .finite_field import FiniteField as _FiniteField
from .binary_finite_field import BinaryFiniteField
from .base_numbers import BaseNumber, DIGITS, BaseNumberTypeError, BaseNumberValueError


class FiniteField:
    """Unifying BinaryFiniteField and FiniteField under one class"""

    def __init__(self, a: int, p: int) -> None:
        if p in PRIMES:
            self.__class__ = _FiniteField  # type: ignore
            self.__init__(a, p)  # type: ignore
        elif p in POWERS_2:
            self.__class__ = BinaryFiniteField  # type: ignore
            self.__init__(a, p)  # type: ignore
        else:
            raise ValueError(f"{p} is neither of prime {PRIMES} or prime factor of 2 {POWERS_2}")
