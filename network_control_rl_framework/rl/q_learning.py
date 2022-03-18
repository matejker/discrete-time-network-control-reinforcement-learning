import numpy as np
from typing import Any, Optional, Tuple, Dict, Union, List

from network_control_rl_framework.rl.model import RLModel
from network_control_rl_framework.algebra import BaseNumber
from network_control_rl_framework.rl.policy import random_action
from network_control_rl_framework.progress_bar import progress_bar_simple
from network_control_rl_framework.network import Network


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

    def train(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)

        coef = max(self.n // 10, 1)
        progress_bar_simple(0, self.num_episodes, prefix="Training:", suffix="Complete", length=50)

        for eps in range(self.num_episodes):
            state = self.initial_state
            action = BaseNumber(self.m, self.q)

            for t in range(self.max_iteration):
                value = self.q_dict.get((state.a, action.a), 0.1)
                next_action, next_state, max_value = self.get_best_action_for_state(state)

                # Explore
                if np.random.rand() < self.epsilon:
                    next_action: BaseNumber = random_action(self.q, self.m)  # type: ignore

                reward = (next_state == self.end_state) * 1
                self.q_dict[state.a, action.a] = min(value + self.alpha * (reward + self.gamma * max_value - value), 1)

                state = next_state
                action = next_action

                if reward:
                    if t + 1 < self.time_horizon:
                        self.time_horizon = t + 1
                    break

            if eps % coef == 0:
                progress_bar_simple(eps, self.num_episodes, prefix="Progress:", suffix="Complete", length=50)
        progress_bar_simple(self.num_episodes, self.num_episodes, prefix="Progress:", suffix="Complete", length=50)

    def get_signals(self, vector: bool = False) -> Union[List[BaseNumber], np.ndarray]:
        if len(self.q_dict) < 2:
            raise ValueError("Tabular values Q are empty, you need to train() the model first.")

        # TODO: rewrite it into np.ndarray
        states: List[BaseNumber] = [self.initial_state]
        signals = []
        if vector:
            u = np.zeros((self.time_horizon, self.m), dtype=np.int8)

        for t in range(self.time_horizon):
            action, state, _ = self.get_best_action_for_state(states[t])
            states.append(state)
            signals.append(action)

            if vector:
                u[t] = action.to_array()

        if vector:
            return u
        else:
            return signals
