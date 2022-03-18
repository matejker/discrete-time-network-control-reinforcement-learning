import numpy as np
from typing import Optional, Dict

from network_control_rl_framework.network import Network
from network_control_rl_framework.rl.model import RLModel
from network_control_rl_framework.algebra import BaseNumber
from network_control_rl_framework.rl.policy import random_action

INF = float("inf")


class Sarsa(RLModel):
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
        n_steps: int = 1,
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
        self.n_steps = n_steps

        self.q_dict: Dict[Any, Any] = {-1: {}}  # TODO: fix typing
        self.all_possible_action = np.arange(self.q ** self.m)  # TODO: can we do better?

    def train(self):
        pass
