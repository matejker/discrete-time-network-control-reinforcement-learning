import numpy as np
from math import log2

from .primes import POWERS_2, PRIMES

POSSIBLE_BASES = set(PRIMES + POWERS_2)
DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]


# TODO: think about hex()
# https://www.journaldev.com/22902/python-hex
class BaseNumber:
    def __init__(self, n: int, q: int, a: int = 0) -> None:
        if q not in POSSIBLE_BASES:
            raise ValueError(f"Cannot evaluate finite field of order {q}, expected value {POSSIBLE_BASES}")

        if a > 0 and n < log2(a) / log2(q):
            raise ValueError(
                f"Number {a} in {q} base is larger than number of position {n}, max possible value {q ** n - 1}"
            )

        self.a: int = a
        self.q: int = q
        self.n: int = n

    @staticmethod
    def convert_array_to_decimal(array: np.ndarray, q: int):
        return int("".join([DIGITS[int(i)] for i in array]), q)

    def from_array(self, array: np.ndarray):
        self.n = len(array)
        self.a = self.convert_array_to_decimal(array, self.q)

    def to_array(self) -> np.ndarray:
        array = np.zeros(self.n, dtype=np.int8)
        temp_number = self.a

        for i in reversed(range(self.n)):
            if temp_number > 0:
                array[i] = temp_number % self.q
                temp_number = temp_number // self.q
            else:
                break

        return array

    def to_string(self) -> str:
        return "".join([DIGITS[i] for i in self.to_array()])

    def __add__(self, other):
        if self.q != other.q:
            raise TypeError(f"The bases aren't the same {self.q=}!={other.q=}")

        if self.n != other.n:
            raise TypeError(f"The lengths aren't the same {self.n=}!={other.n=}")

        a = self.to_array()
        b = other.to_array()
        return BaseNumber(self.n, self.q, self.convert_array_to_decimal((a + b) % self.q, self.q))

    def __sub__(self, other):
        if self.q != other.q:
            raise TypeError(f"The bases aren't the same {self.q=}!={other.q}")

        if self.n != other.n:
            raise TypeError(f"The lengths aren't the same {self.n=}!={other.n}")

        a = self.to_array()
        b = other.to_array()
        return BaseNumber(self.n, self.q, self.convert_array_to_decimal((a - b) % self.q, self.q))

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            return self.a == other
        elif hasattr(other, "a"):
            return self.a == other.a
        return False

    def __ne__(self, other) -> bool:
        return not self == other

    def __repr__(self) -> str:
        return f"BaseNumber(a={self.to_string()}_{self.q}={self.a}_10, n={self.n}, q={self.q})"

    def __str__(self) -> str:
        return str(self.a)
