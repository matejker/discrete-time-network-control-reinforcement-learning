from .primes import PRIMES


class FiniteField:
    def __init__(self, a: int, n: int) -> None:
        if n not in PRIMES:
            raise ValueError(f"{n} is either not a prime or it is too high, only {PRIMES} are allowed!")

        self.a = a
        self.n = n

    def __add__(self, other: FiniteField) -> FiniteField:
        return FiniteField((self.a + other.a) % self.n, self.n)

    def __pow__(self, power: int) -> FiniteField:
        return FiniteField((self.a ^ power) % self.n, self.n)

    def __mul__(self, other: FiniteField) -> FiniteField:
        return FiniteField((self.a * other.a) % self.n, self.n)

    def __sub__(self, other: FiniteField) -> FiniteField:
        return FiniteField((self.a - other.a) % self.n, self.n)

    def __repr__(self):
        return str(self.a)

    def __str__(self):
        return f"FiniteField({self.a}, {self.n}) Z_{self.n}"
