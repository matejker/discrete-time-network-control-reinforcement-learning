from typing import Optional, Dict

from network_control_rl_framework.network import Network
from network_control_rl_framework.algebra import BaseNumber


class RLModel:
    def __init__(
        self,
        initial_state: BaseNumber,
        end_state: BaseNumber,
        network: Network,
        input_matrix: Dict[int, int],
        num_episodes: Optional[int] = None,
        episodes_factor: Optional[float] = None,
        max_iteration: Optional[int] = None,
        iteration_factor: Optional[float] = None,
    ) -> None:
        self.initial_state = initial_state
        self.end_state = end_state
        self.input_matrix = input_matrix
        self.m: int = len(input_matrix)  # Number of driver nodes

        if initial_state.q != end_state.q:
            raise ValueError(
                f"Base / Finite Field order has to be the same for both "
                f"initial and end state, {initial_state.q}!={end_state.q}"
            )

        self.network = network
        self.time_horizon = network.nodes

        if network.nodes != initial_state.n or initial_state.n != end_state.n or end_state.n != network.nodes:
            raise ValueError(
                f"Size of network, initial and end state has to be the same, "
                f"{network.nodes=}, {initial_state.n=}, {end_state.n=}"
            )

        self.q = initial_state.q
        self.n = initial_state.n

        if iteration_factor:
            self.max_iteration = int(network.nodes * iteration_factor)

        if max_iteration:
            self.max_iteration = max_iteration

        if not self.max_iteration:
            raise ValueError()

        if episodes_factor:
            self.num_episodes = int(network.nodes * episodes_factor)

        if num_episodes:
            self.num_episodes = num_episodes

        if not self.num_episodes:
            raise ValueError()

    def train(self):
        pass

    def __repr__(self):
        pass

    def __str__(self):
        pass