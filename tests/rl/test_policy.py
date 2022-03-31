import numpy as np
from pytest import raises

from network_control_rl.rl import random_action
from network_control_rl.algebra import BaseNumber


def test_random_action_q_n():
    q = 3
    n = 4
    random_action_base_number = random_action(q, n, seed=16)
    assert random_action_base_number.a == int("1211", 3)

    random_action_vector = random_action(q, n, vector=True, seed=16)
    assert (random_action_vector == np.array([1, 2, 1, 1])).all()


def test_random_action_base_number():
    q = 3
    n = 4
    number = BaseNumber(n, q)
    random_action_base_number = random_action(base=number, seed=16)
    assert random_action_base_number.a == int("1211", 3)

    random_action_vector = random_action(base=number, vector=True, seed=16)
    assert (random_action_vector == np.array([1, 2, 1, 1])).all()


def test_random_action_exceptions():
    with raises(ValueError):
        # Either `q` and `n` or `base` needs to be specify.
        random_action(1)
