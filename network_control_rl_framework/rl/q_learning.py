import numpy as np
from typing import Optional

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
        self.q_dict = {}

    def train(self):
        random_explore_vec = np.random.rand(size=self.num_episodes) < self.epsilon
        for t in range(self.num_episodes):
            # Explore
            if random_explore_vec[t]:
                next_action: BaseNumber = random_action(self.m, self.q)

            # Exploit
            else:
                pass
