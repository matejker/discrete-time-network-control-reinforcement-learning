from math import log2

from .primes import IRR_POLYNOMIALS, POWERS_2


class BinaryFiniteField:
    """Basic arithmetic operations on finite fields or Galois field of order 2^n
    Params:
        - a (int) element of the 2^n set
        - p (int) prime factor
        - r (int): irreducible polynomial over GF(2^n)
    Example:
        GF(2^2):
        Tables for addition and multiplication
          + | 0   | 1   | x   | x+1 |     * | 0   | 1   | x   | x+1 |
        ----+-----+-----+-----+-----+   ----+-----+-----+-----+-----+
          0 | 0   | 1   | x   | x+1 |     0 | 0   | 0   | 0   | 0   |
        ----+-----+-----+-----+-----+   ----+-----+-----+-----+-----+
          1 | 1   | 0   | x+1 | x   |     1 | 0   | 1   | x   | x+1 |
        ----+-----+-----+-----+-----+   ----+-----+-----+-----+-----+
          x | x   | x+1 | 0   | 1   |     x | 0   | x   | x+1 | 1   |
        ----+-----+-----+-----+-----+   ----+-----+-----+-----+-----+
         x+1| x+1 | x   | 1   | 0   |    x+1| 0   | x+1 | 1   | x   |

         >> a = BinaryFiniteField(2, 2) # x
         >> b = BinaryFiniteField(3, 2) # x+1
         >> a + b # 1
         >> a * b # 1
         >> a / b # x+1
         >> ~b  # (inverse of b) 2
    References:
        [1] John Kerl, (2004), "Computation in finite fields", https://johnkerl.org/doc/ffcomp.pdf
    """

    def __init__(self, a: int, p: int, r: int = None) -> None:
        self.a = a
        self.p = p
        self.n = int(log2(p))

        if p not in POWERS_2:
            raise ValueError(f"Cannot evaluate finite field of order {p}, expected value {POWERS_2}")

        self.r = r or IRR_POLYNOMIALS[self.n - 1]

    def print_polynomial(self) -> str:
        b = bin(self.r)[2:]
        return "+".join([f"x^{len(b) - k - 1}" for k, i in enumerate(b) if int(i) == 1]).replace("x^0", "1")

    def get_polynomial_degree(self) -> int:
        if self.r == 0:
            return 0
        degree = 0
        temp_r = self.r
        while temp_r >> 1:
            temp_r >>= 1
            degree += 1

        return degree

    def __add__(self, other):
        return BinaryFiniteField(self.a ^ other.a, self.p)

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        x = self.a - other.a

        if x < 0:
            x = self.n - x - 1
        return BinaryFiniteField(x, self.n)

    def __mul__(self, other):
        degree_r = self.get_polynomial_degree() - 1
        prod = 0
        temp_a = self.a
        temp_b = other.a

        while temp_a and temp_b:
            if temp_b & 1:
                prod ^= temp_a

            temp_b >>= 1

            if temp_a & (1 << degree_r):
                temp_a = (temp_a << 1) ^ self.r
            else:
                temp_a <<= 1

        return BinaryFiniteField(prod, self.p)

    def __pow__(self, power: int):
        a2 = self
        prod = BinaryFiniteField(1, self.p)

        while power != 0:
            if power & 1:
                prod = prod * a2
            power = power >> 1
            a2 = a2 * a2

        return prod

    def __invert__(self):
        degr = self.get_polynomial_degree()
        power = (1 << degr) - 2
        return self**power

    def __truediv__(self, other):
        binv = ~other
        return self * binv

    def __repr__(self):
        return str(self.a)

    def __str__(self):
        return f"BinaryFiniteField({self.a}, {self.n}) GF(2^{self.n}) ≃ Z_2 / <{self.print_polynomial()}>"

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            return self.a == other
        elif hasattr(other, "a"):
            return self.a == other.a
        return False

    def __ne__(self, other) -> bool:
        return not self == other
