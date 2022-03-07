import numpy as np
from typing import Optional
from itertools import combinations, product

from .network import Network
from network_control_rl_framework.utils import random_choice


def erdos_renyi_random_network(
    n: int, p: float = 0.1, seed: Optional[int] = None, ignore_self_loops: bool = True
) -> Network:
    """Erdos-Renyi random network [1]
    Args:
        - n (integer): number of network nodes
        - p (float): probability of link existence
        - seed=None (integer): numpy.random.seed (integer between 0 and 2**32 - 1 inclusive) [1]
    Returns:
        A Network object.
    References:
        [1] Newman, M. E. J. (2010), Networks: an introduction,
        Oxford University Press, Oxford; New York
    """
    if seed:
        np.random.seed(seed)
    edges = [
        e for e in product(range(n), repeat=2) if np.random.random() < p and (e[0] != e[1] or not ignore_self_loops)
    ]

    return Network(edges=edges, n=n)


def barabasi_albert_preferential_attachment_network(
    n: int, m: int, m0: Optional[int] = None, seed: Optional[int] = None
) -> Network:
    """Barabasi-Albert model [2], generate a random choice of given size with no repeating.
    Args:
        - n (integer): number of network nodes
        - m (integer): number of new edges from newly added node
        - m0=None (integer): size of initial connected network
        - seed=None (integer): numpy.random.seed (integer between 0 and 2**32 - 1 inclusive) [1]
    Returns:
        A Network object.
    Raises:
        ValueError: If inserted m <= m0
        ValueError: If inserted n => m0
    References:
        [1] The SciPy community, Numpy.random.seed,
        https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.seed.html
        [2] Newman, M. E. J. (2010), Networks: an introduction,
        Oxford University Press, Oxford; New York
    """
    if seed:
        np.random.seed(seed)

    m0 = m0 or m

    if m0 < m:
        raise ValueError(f"Inserted values are not correct, m <= m0, m={m} and m0={m0}")

    if n < m0:
        raise ValueError(f"Inserted values are not correct, n => m0, m0={m0} and n={n}")

    # Initial complete graph
    edges = list(combinations(list(range(m0)), 2))
    nodes_basket = list(range(m0)) * (m0 - 1)

    for r in range(m0, n):
        connections = random_choice(nodes_basket, m)
        rm = [r] * m
        for c in connections:
            if np.random.random() < 0.5:
                edges.append((c, r))
            else:
                edges.append((r, c))
        nodes_basket.extend(rm + connections)

    return Network(edges=edges, n=n)
