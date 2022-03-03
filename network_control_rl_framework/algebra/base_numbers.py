import numpy as np

from .primes import POWERS_2, PRIMES

POSSIBLE_BASES = set(PRIMES + POWERS_2)
DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f"]


# TODO: think about hex()
# https://www.journaldev.com/22902/python-hex
class BaseNumber:
    def __init__(self, n: int, q: int, a: int = 0) -> None:
        self.a: int = a
        self.q: int = q
        self.n: int = n

    @staticmethod
    def convert_array_to_decimal(array: np.ndarray, q: int):
        return int("".join([DIGITS[i] for i in array]), q)

    def from_array(self, array: np.ndarray):
        self.n = len(array)
        self.a = self.convert_array_to_decimal(array, self.q)

    def to_array(self) -> np.ndarray:
        array = np.zeros(self.n)
        temp_number = self.a

        for i in reversed(range(self.n)):
            if temp_number > 0:
                array[i] = temp_number % self.q
                temp_number = temp_number // self.q
            else:
                break

        return array

    def __add__(self, other):
        if self.q != other.q:
            raise TypeError(f"The bases aren't the same {self.q=}!={other.q}")

        if self.n != other.n:
            raise TypeError(f"The lengths aren't the same {self.n=}!={other.n}")

        a = self.to_array()
        b = other.to_array()
        return BaseNumber(self.n, self.q, self.convert_array_to_decimal((a + b) % self.q))

    def __sub__(self, other):
        if self.q != other.q:
            raise TypeError(f"The bases aren't the same {self.q=}!={other.q}")

        if self.n != other.n:
            raise TypeError(f"The lengths aren't the same {self.n=}!={other.n}")

        a = self.to_array()
        b = other.to_array()
        return BaseNumber(self.n, self.q, self.convert_array_to_decimal((a - b) % self.q))
