from .primes import PRIMES


class FiniteField:
    """Basic arithmetic operations on finite fields over a prime
    Params:
        - a (int) element
        - p (int) prime

    Example:
        GF(p):

         >> a = FiniteField(2, 5)
         >> b = FiniteField(3, 5)
         >> a + b # 0
         >> a * b # 1
         >> a / b # 1
         >> ~b  # (inverse of b) 2
    """

    def __init__(self, a: int, p: int) -> None:
        if p not in PRIMES:
            raise ValueError(f"{p} is either not a prime or it is too high, only {PRIMES} are allowed!")

        self.a = a % p  # If 'a' is bigger than p
        self.p = p

    def __add__(self, other):
        return FiniteField((self.a + other.a) % self.p, self.p)

    def __pow__(self, power: int):
        return FiniteField((self.a ^ power) % self.p, self.p)

    def __mul__(self, other):
        return FiniteField((self.a * other.a) % self.p, self.p)

    def __sub__(self, other):
        return FiniteField((self.a - other.a) % self.p, self.p)

    def __invert__(self):
        # However, a^(p^n - 1) = 1 in GF(p), therefore, inverse of a is a^(p^n - 2)
        return FiniteField((self.a ** (self.p - 2)) % self.p, self.p)

    def __truediv__(self, other):
        binv = ~other
        return self * binv

    def __repr__(self):
        return str(self.a)

    def __str__(self):
        return f"FiniteField({self.a}, {self.n}) Z_{self.n}"
