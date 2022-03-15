import numpy as np
from pytest import raises

from network_control_rl_framework.rl import RLModel
from network_control_rl_framework.algebra import BaseNumber


def test_rl_model_object_init(network_cycle_4):
    input_matrix = {0: 0}
    q = 4
    n = network_cycle_4.nodes

    initial_state = BaseNumber(n, q)
    initial_state.from_array(np.array([1, 2, 3, 1]))
    end_state = BaseNumber(n, q)
    end_state.from_array(np.array([1, 3, 2, 1]))

    model = RLModel(initial_state, end_state, network_cycle_4, input_matrix, episodes_factor=13, iteration_factor=2)

    assert model.q == q
    assert model.n == n
    assert model.m == 1
    assert model.max_iteration == 8
    assert model.num_episodes == 52


def test_rl_model_object_init_exception(network_cycle_4):
    input_matrix = {0: 0}
    q = 4
    n = network_cycle_4.nodes

    # Bases don't match
    initial_state = BaseNumber(n, q - 1)
    initial_state.from_array(np.array([1, 2, 0, 1]))
    end_state = BaseNumber(n, q)
    end_state.from_array(np.array([1, 3, 2, 1]))

    with raises(ValueError):
        RLModel(initial_state, end_state, network_cycle_4, input_matrix, episodes_factor=13, iteration_factor=2)

    # Sizes don't match
    initial_state = BaseNumber(n, q)
    initial_state.from_array(np.array([1, 2, 0, 1]))
    end_state = BaseNumber(n + 1, q)
    end_state.from_array(np.array([1, 3, 2, 1, 2]))

    with raises(ValueError):
        RLModel(initial_state, end_state, network_cycle_4, input_matrix, episodes_factor=13, iteration_factor=2)
