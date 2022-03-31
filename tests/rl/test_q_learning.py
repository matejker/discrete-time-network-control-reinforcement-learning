import numpy as np
from pytest import raises

from network_control_rl.rl import QLearning
from network_control_rl.algebra import BaseNumber


def test_q_learning_training(network_cycle_4):
    input_matrix = {0: 0}
    q = 4
    n = network_cycle_4.nodes

    initial_state = BaseNumber(n, q)
    initial_state.from_array(np.array([1, 2, 3, 1]))
    end_state = BaseNumber(n, q)
    end_state.from_array(np.array([1, 3, 2, 1]))

    model = QLearning(initial_state, end_state, network_cycle_4, input_matrix, episodes_factor=13, iteration_factor=2)

    assert model.__repr__() == "QLearning(trained=not yet)"
    assert not model.is_trained()

    with raises(ValueError):
        model.get_signals()

    model.train(seed=6)

    assert model.__repr__() == "QLearning(trained=yes, time_horizon=3)"
    assert model.is_trained()

    action_t1 = BaseNumber(1, q, 3)
    action_t2 = BaseNumber(1, q, 0)
    action_t3 = BaseNumber(1, q, 3)

    assert model.get_signals() == [action_t1, action_t2, action_t3]
    assert (model.get_signals(vector=True) == np.array([[3], [0], [3]])).all()

    a1, s1, _ = model.get_best_action_for_state(initial_state)
    a2, s2, _ = model.get_best_action_for_state(s1)
    a3, s3, _ = model.get_best_action_for_state(s2)

    assert s1 == BaseNumber(n, q, 155)
    assert s2.a == BaseNumber(n, q, 230)
    assert s3 == end_state

    assert a1 == action_t1
    assert a2 == action_t2
    assert a3 == action_t3
