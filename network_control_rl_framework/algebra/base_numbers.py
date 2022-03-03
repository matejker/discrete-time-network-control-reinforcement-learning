import numpy as np

from .primes import POWERS_2, PRIMES

POSSIBLE_BASES = set(PRIMES + POWERS_2)


class BaseNumber:
    def __init__(self, a: int, n: int) -> None:
        self.a = a
        self.n = n

    def from_array(self, array: np.ndarray):
        pass

    def to_array(self, number: int):
        pass

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass
