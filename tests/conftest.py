import pytest

from network_control_rl_framework.network import Network


@pytest.fixture()
def network_cycle_4():
    network = Network()
    network.from_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    return network
