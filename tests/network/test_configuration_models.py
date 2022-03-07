import pytest

from network_control_rl_framework.network import (
    Network,
    erdos_renyi_random_network,
    barabasi_albert_preferential_attachment_network,
)


def test_erdos_renyi_random_network():
    network = erdos_renyi_random_network(n=5, p=0.5, seed=16)
    assert type(network) == Network
    assert network.edges == {
        (2, 4),
        (1, 2),
        (0, 4),
        (2, 1),
        (3, 4),
        (4, 0),
        (4, 3),
        (0, 3),
        (3, 0),
        (2, 3),
        (1, 0),
        (3, 2),
        (1, 3),
    }
    assert network.nodes == 5
    assert network.edge_basket == {0: [3, 4], 1: [0, 2, 3], 2: [1, 3, 4], 3: [0, 2, 4], 4: [0, 3]}

    network_self_loops = erdos_renyi_random_network(n=3, p=0.5, seed=16, ignore_self_loops=False)
    assert network_self_loops.edges == {(1, 2), (2, 1), (0, 0), (1, 1), (2, 2), (1, 0)}
    assert network_self_loops.nodes == 3
    assert network_self_loops.edge_basket == {0: [0], 1: [0, 1, 2], 2: [1, 2]}


def test_barabasi_albert_preferential_attachment_network():
    network = barabasi_albert_preferential_attachment_network(n=5, m=2, m0=2, seed=16)

    assert type(network) == Network
    assert network.edges == {(0, 1), (4, 0), (2, 1), (0, 3), (2, 0), (4, 2), (1, 3)}
    assert network.nodes == 5
    assert network.edge_basket == {0: [1, 3], 1: [3], 2: [0, 1], 4: [0, 2]}


def test_barabasi_albert_preferential_attachment_network_incorrect_parameters():
    with pytest.raises(ValueError):
        barabasi_albert_preferential_attachment_network(n=5, m=2, m0=1, seed=16)

    with pytest.raises(ValueError):
        barabasi_albert_preferential_attachment_network(n=1, m=2, m0=2, seed=16)
