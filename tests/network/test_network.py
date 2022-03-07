import pytest

from network_control_rl_framework.network import Network, WeightNetwork


def test_network():
    network = Network()
    assert network.edges == set()
    assert network.nodes == 0
    assert network.edge_basket == dict()

    network.from_edges([(1, 2), (0, 3), (5, 4)])
    assert network.nodes == 5
    assert network.edges == {(1, 2), (0, 3), (5, 4)}
    assert network.edge_basket == {1: [2], 0: [3], 5: [4]}


def test_weight_network():
    network = WeightNetwork()
    assert network.edges == set()
    assert network.nodes == 0
    assert network.edge_basket == dict()

    network.from_edges([(1, 2, 0.5), (0, 3, 0.1), (5, 4, 20)])
    assert network.nodes == 5
    assert network.edges == {(1, 2, 0.5), (0, 3, 0.1), (5, 4, 20)}
    assert network.edge_basket == {1: [(2, 0.5)], 0: [(3, 0.1)], 5: [(4, 20)]}


def test_network_non_int_nodes():
    with pytest.raises(ValueError):
        Network(edges=[("lorem", 2)])


def test_weight_network_not_float():
    with pytest.raises(ValueError):
        WeightNetwork(edges=[(3, 2, "ipsum")])
