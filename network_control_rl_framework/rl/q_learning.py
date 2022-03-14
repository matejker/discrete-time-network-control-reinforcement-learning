import numpy as np
from typing import Any, Optional, Tuple, Dict

from network_control_rl_framework.network import Network
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

    def get_best_action_for_state(self, state: BaseNumber, seed: Optional[int] = None) -> Tuple[int, float]:
        if seed:
            np.random.seed(seed)

        max_value: float = -1.0
        best_action: int = -1

        already_explored_actions = self.q_dict.get(state, {})
        for action, value in already_explored_actions.items():
            if value > max_value:
                max_value = value
                best_action = action

        return best_action, max_value

    def train(self, seed: Optional[int] = None):
        random_explore_vec = np.random.rand(size=self.num_episodes) < self.epsilon
        for t in range(self.num_episodes):
            # Explore
            if random_explore_vec[t]:
                next_action: BaseNumber = random_action(self.m, self.q)

            # Exploit
            else:
                pass
