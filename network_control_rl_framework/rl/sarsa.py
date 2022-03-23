import numpy as np
from typing import Optional, Dict

from network_control_rl_framework.rl.model import RLModel, DEFAULT_VALUE
from network_control_rl_framework.algebra import BaseNumber
from network_control_rl_framework.rl.policy import random_action
from network_control_rl_framework.progress_bar import progress_bar_simple
from network_control_rl_framework.network import Network, calculate_next_state_base_number


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

    def train(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)

        coef = max(self.n // 10, 1)
        progress_bar_simple(0, self.num_episodes, prefix="Training:", suffix="Complete", length=50)

        for eps in range(self.num_episodes):
            state = self.initial_state
            action = BaseNumber(self.m, self.q)
            time = self.max_iteration

            value = self.q_dict.get((state.a, action.a), DEFAULT_VALUE)
            states = [self.initial_state]
            actions = [BaseNumber(self.m, self.q)]

            for t in range(self.max_iteration):
                if t < time:
                    # Explore
                    if np.random.rand() < self.epsilon:
                        next_action: BaseNumber = random_action(self.q, self.m)  # type: ignore
                        next_state = calculate_next_state_base_number(
                            self.network, state, next_action, self.input_matrix
                        )
                        next_value = self.q_dict.get((next_state.a, next_action.a), DEFAULT_VALUE)

                    # Exploit
                    else:
                        next_action, next_state, next_value = self.get_best_action_for_state(state)

                    reward = (next_state == self.end_state) * 1
                    self.q_dict[state.a, action.a] = min(
                        value + self.alpha * (reward + self.gamma * next_value - value), DEFAULT_VALUE
                    )

                    state = next_state
                    action = next_action
                    value = next_value

                    actions.append(action)
                    states.append(state)

                    if reward:
                        if t + 1 < self.time_horizon:
                            self.time_horizon = t + 1
                            time = t
                        break

                tau = t - self.n_steps
                if tau >= 0:
                    returns = pow(self.gamma, min(time - tau - 1, self.n_steps - 1))

                    if tau + self.n_steps <= time:
                        returns += pow(self.gamma, self.n_steps) + self.q_dict.get(
                            (states[tau - self.n_steps].a, actions[tau - self.n_steps].a), DEFAULT_VALUE
                        )

                    if states[tau] == self.end_state:
                        value = self.q_dict.get((states[tau].a, actions[tau].a), DEFAULT_VALUE)
                        self.q_dict[states[tau].a, actions[tau].a] = min(
                            value + self.alpha * (returns - value), DEFAULT_VALUE
                        )

            if eps % coef == 0:
                progress_bar_simple(eps, self.num_episodes, prefix="Progress:", suffix="Complete", length=50)
        progress_bar_simple(self.num_episodes, self.num_episodes, prefix="Progress:", suffix="Complete", length=50)
