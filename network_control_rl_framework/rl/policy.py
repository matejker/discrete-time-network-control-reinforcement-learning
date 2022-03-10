import numpy as np
from typing import Optional, Union

from network_control_rl_framework.algebra import DIGITS, BaseNumber


def random_action(
    q: Optional[int] = None,
    n: Optional[int] = None,
    base: Optional[BaseNumber] = None,
    vector: bool = False,
    seed: Optional[int] = None,
) -> Union[BaseNumber, np.ndarray]:
    if seed:
        np.random.seed(seed)

    if not (q and n) and not base:
        raise ValueError("Either `q` and `n` or `base` needs to be specify.")

    if not (q or n) and base:
        n = base.n
        q = base.q

    random_integers = np.random.randint(0, q, size=n)
    if vector:
        return random_integers
    number = "".join([DIGITS[i] for i in random_integers])
    return BaseNumber(n, q, int(number, q))  # type: ignore
