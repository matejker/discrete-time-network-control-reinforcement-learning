import numpy as np
from typing import Optional, Dict

from network_control_rl_framework.rl.model import RLModel, DEFAULT_VALUE
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

    def train(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)

        coef = max(self.n // 10, 1)
        progress_bar_simple(0, self.num_episodes, prefix="Training:", suffix="Complete", length=50)

        for eps in range(self.num_episodes):
            state = self.initial_state
            action = BaseNumber(self.m, self.q)

            for t in range(self.max_iteration):
                value = self.q_dict.get((state.a, action.a), DEFAULT_VALUE)
                next_action, next_state, max_value = self.get_best_action_for_state(state)

                # Explore
                if np.random.rand() < self.epsilon:
                    next_action: BaseNumber = random_action(self.q, self.m)  # type: ignore

                reward = (next_state == self.end_state) * 1
                self.q_dict[state.a, action.a] = min(
                    value + self.alpha * (reward + self.gamma * max_value - value), DEFAULT_VALUE
                )

                state = next_state
                action = next_action

                if reward:
                    if t + 1 < self.time_horizon:
                        self.time_horizon = t + 1
                    break

            if eps % coef == 0:
                progress_bar_simple(eps, self.num_episodes, prefix="Progress:", suffix="Complete", length=50)
        progress_bar_simple(self.num_episodes, self.num_episodes, prefix="Progress:", suffix="Complete", length=50)

    def __repr__(self) -> str:
        trained = f"yes, time_horizon={self.time_horizon}" if self.is_trained() else "not yet"
        return f"QLearning(trained={trained})"
