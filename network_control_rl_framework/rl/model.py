import numpy as np
from typing import Any, Optional, Dict, Tuple, Union, List

from network_control_rl_framework.algebra import BaseNumber
from network_control_rl_framework.network import Network, calculate_next_state_base_number

DEFAULT_VALUE = 0.1


class RLModelValueError(ValueError):
    pass


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
            raise RLModelValueError(
                f"Base / Finite Field order has to be the same for both "
                f"initial and end state, {initial_state.q}!={end_state.q}"
            )

        self.network = network
        self.time_horizon = float("INF")

        if network.nodes != initial_state.n or initial_state.n != end_state.n or end_state.n != network.nodes:
            raise RLModelValueError(
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

        self.q_dict: Dict[Any, Any] = {-1: {}}  # TODO: fix typing
        self.all_possible_action = np.arange(self.q**self.m)  # TODO: can we do better?

    def get_best_action_for_state(self, state: BaseNumber) -> Tuple[BaseNumber, BaseNumber, float]:
        max_value: float = -1.0
        best_action = BaseNumber(self.m, self.q)
        next_state = state

        np.random.shuffle(self.all_possible_action)
        for action in self.all_possible_action:
            action_base = BaseNumber(self.m, self.q, action)
            temp_state = calculate_next_state_base_number(self.network, state, action_base, self.input_matrix)
            value = self.q_dict.get((temp_state.a, action), DEFAULT_VALUE)

            if value > max_value:
                max_value = value
                best_action = action_base
                next_state = temp_state

        return best_action, next_state, max_value

    def is_trained(self) -> bool:
        return self.time_horizon <= self.network.nodes

    def train(self):
        pass

    def __repr__(self) -> str:
        trained = f"yes, time_horizon={self.time_horizon}" if self.is_trained() else "not yet"
        return f"RLModel(trained={trained})"

    def get_signals(self, vector: bool = False) -> Union[List[BaseNumber], np.ndarray]:
        if len(self.q_dict) < 2:
            raise ValueError("Tabular values Q are empty, you need to train() the model first.")

        # TODO: rewrite it into np.ndarray
        states: List[BaseNumber] = [self.initial_state]
        signals = []
        time_horizon = int(min(self.network.nodes, self.time_horizon))
        if vector:
            u = np.zeros((time_horizon, self.m), dtype=np.int8)

        for t in range(time_horizon):
            action, state, _ = self.get_best_action_for_state(states[t])
            states.append(state)
            signals.append(action)

            if vector:
                u[t] = action.to_array()

        if vector:
            return u
        else:
            return signals
