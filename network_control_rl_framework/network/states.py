import numpy as np
from typing import Dict

from network_control_rl_framework.network import Network
from network_control_rl_framework.algebra import BaseNumber, FiniteField

""" Modular Arithmetic and the Distributive Property
    (a + b) mod c = ((a mod c) + (b mod c)) mod c
"""


def calculate_next_state(
    network: Network, x: np.ndarray, signals: np.ndarray, input_matrix: Dict[int, int], q: int
) -> np.ndarray:
    n = len(x)

    if n != network.nodes:
        raise ValueError(f"Length of state vector x and number of nodes doesn't match, {n}!={network.nodes}")

    new_x = np.zeros(n, dtype=np.int8)
    for node, neighbours in network.edge_basket.items():
        for neighbour in neighbours:
            new_x[neighbour] = (FiniteField(x[node], q) + FiniteField(new_x[neighbour], q)).a

    for i, b in enumerate(signals):
        new_x[input_matrix[i]] = (FiniteField(new_x[input_matrix[i]], q) + FiniteField(b, q)).a

    return new_x


def calculate_next_state_base_number(
    network: Network, number: BaseNumber, signals: BaseNumber, input_matrix: Dict[int, int]
) -> BaseNumber:
    n = number.n
    q = number.q
    x = number.to_array()

    if n != network.nodes:
        raise ValueError(f"Length of state vector x and number of nodes doesn't match, {n}!={network.nodes}")

    new_x = np.zeros(n, dtype=np.int8)
    for node, neighbours in network.edge_basket.items():
        for neighbour in neighbours:
            new_x[neighbour] = (FiniteField(x[node], q) + FiniteField(new_x[neighbour], q)).a

    for i, b in enumerate(signals):
        new_x[input_matrix[i]] = (FiniteField(new_x[input_matrix[i]], q) + FiniteField(b, q)).a

    new_number = BaseNumber(n, q)
    new_number.from_array(new_x)
    return new_number
