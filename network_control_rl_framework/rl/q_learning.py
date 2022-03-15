import numpy as np
from typing import Any, Optional, Tuple, Dict, Union, List

from network_control_rl_framework.network import Network, calculate_next_state_base_number
from network_control_rl_framework.rl.model import RLModel
from network_control_rl_framework.rl.policy import random_action
from network_control_rl_framework.algebra import BaseNumber


class QLearning(RLModel):
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
        epsilon: float = 0.01,
        gamma: float = 0.99,
        alpha: float = 0.5,
    ):
        RLModel.__init__(
            self,
            initial_state,
            end_state,
            network,
            input_matrix,
            num_episodes,
            episodes_factor,
            max_iteration,
            iteration_factor,
        )

        # TODO: add docs
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.q_dict: Dict[Any, Any] = {-1: {}}  # TODO: fix typing
        self.all_possible_action = np.arange(self.q**self.m)  # TODO: can we do better?

    def get_best_action_for_state(self, state: BaseNumber) -> Tuple[BaseNumber, BaseNumber, float]:
        max_value: float = -1.0
        best_action: int = -1
        next_state = state

        np.random.shuffle(self.all_possible_action)
        for action in self.all_possible_action:
            temp_state = calculate_next_state_base_number(self.network, state, action, self.input_matrix)
            value = self.q_dict.get(temp_state, {}).get(action, 0.1)

            if value > max_value:
                max_value = value
                best_action = action
                next_state = temp_state

        return BaseNumber(self.m, self.q, best_action), next_state, max_value

    def train(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)

        # TODO: Use progress bar
        for _ in range(self.num_episodes):
            state = self.initial_state
            action = BaseNumber(self.m, self.q)

            for t in range(self.max_iteration):
                value = self.q_dict.get(state, {}).get(action, 0.1)
                next_action, next_state, max_value = self.get_best_action_for_state(state)

                # Explore
                if np.random.rand() < self.epsilon:
                    next_action: BaseNumber = random_action(self.m, self.q)  # type: ignore

                reward = (next_state == self.end_state) * 1
                self.q_dict[state][action] = min(value + self.alpha * (reward + self.gamma * max_value - value), 1)

                state = next_state
                action = next_action

                if reward:
                    if t + 1 < self.time_horizon:
                        self.time_horizon = t + 1
                    break

    def get_signals(self, vector: bool = False) -> Union[List[BaseNumber], np.ndarray]:
        if len(self.q_dict) < 2:
            raise ValueError("Tabular values Q are empty, you need to train() the model first.")

        # TODO: rewrite it into np.ndarray
        states: List[BaseNumber] = [self.initial_state]
        signals = []
        if vector:
            u = np.zeros((self.time_horizon, self.m))

        for t in range(1, self.time_horizon):
            action, state, _ = self.get_best_action_for_state(states[t])
            states.append(state)
            signals.append(action)

            if vector:
                u[t] = action.to_array()

        if vector:
            return u
        else:
            return signals
