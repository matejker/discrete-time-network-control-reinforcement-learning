import numpy as np
from typing import Any


def is_int(number: Any) -> bool:
    try:
        int(number)
        return True
    except (ValueError, TypeError):
        return False


def is_float(number: Any) -> bool:
    try:
        float(number)
        return True
    except (ValueError, TypeError):
        return False


def random_choice(seq, size: int = 2):
    """Generate a random choice of given size with no repeating.
    Args:
        - seq (list): a list of elements to choose from, elements can be repeated
        - size=2 (integer): size of the random set
    Returns:
        Set of random elements of given size
    """
    targets: set = set()

    while len(targets) < size:
        x = np.random.choice(seq)

        targets.add(x)

    return list(targets)
