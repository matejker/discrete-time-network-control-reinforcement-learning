import pytest
import numpy as np

from network_control_rl_framework.algebra import BaseNumber
from network_control_rl_framework.network import calculate_next_state_base_number, calculate_next_state


def test_calculate_next_state(network_cycle_4):
    x = np.array([0, 1, 2, 1], dtype=np.int8)
    signal = np.array([1], dtype=np.int8)
    input_matrix = {0: 1}  # driver node is the node 1

    assert (
        calculate_next_state(network_cycle_4, x, signal, input_matrix, q=3) == np.array([1, 1, 1, 2], dtype=np.int8)
    ).all()


def test_calculate_next_state_base_number(network_cycle_4):
    x = BaseNumber(network_cycle_4.nodes, q=3)
    x.from_array(np.array([0, 1, 2, 1], dtype=np.int8))

    signal = BaseNumber(1, q=3)
    signal.from_array(np.array([1], dtype=np.int8))
    input_matrix = {0: 1}  # driver node is the node 1

    assert calculate_next_state_base_number(network_cycle_4, x, signal, input_matrix).a == int("1112", 3)


def test_calculate_next_state_base_number_size_mismatch(network_cycle_4):
    # network has 4 nodes
    x = BaseNumber(3, q=3)
    x.from_array(np.array([0, 1, 2], dtype=np.int8))  # state vector length is 3

    signal = BaseNumber(1, q=3)
    signal.from_array(np.array([1], dtype=np.int8))
    input_matrix = {0: 1}  # driver node is the node 1
    with pytest.raises(ValueError):
        calculate_next_state_base_number(network_cycle_4, x, signal, input_matrix)


def test_calculate_next_state_size_mismatch(network_cycle_4):
    # network has 4 nodes
    x = np.array([0, 1, 2], dtype=np.int8)  # state vector length is 3
    signal = np.array([1], dtype=np.int8)
    input_matrix = {0: 1}  # driver node is the node 1

    with pytest.raises(ValueError):
        calculate_next_state(network_cycle_4, x, signal, input_matrix, q=3)
