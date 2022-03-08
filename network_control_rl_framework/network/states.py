import numpy as np

from network_control_rl_framework.network import Network
from network_control_rl_framework.algebra import BaseNumber, FiniteField


def calculate_next_state(network: Network, x: np.ndarray) -> np.ndarray:
    pass


def calculate_next_state_base_number(network: Network, number: BaseNumber) -> BaseNumber:
    pass
