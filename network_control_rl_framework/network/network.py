from typing import List, Tuple, Dict
from network_control_rl_framework.utils import is_int, is_float


class Network:
    @staticmethod
    def get_edge_basket(edges: List[Tuple], weighted: bool = False) -> Dict:
        basket: Dict = dict()

        if weighted:
            for i, j, w in edges:
                if not is_int(i) or not is_int(j):
                    raise ValueError(f"Nodes have to be integers, ({i}, {j}) are not")

                if not is_float(w):
                    raise ValueError(f"Weight has to be float, {w} is not")

                basket[i] = basket.get(i, []) + [(j, w)]
        else:
            for i, j in edges:
                if not is_int(i) or not is_int(j):
                    raise ValueError(f"Nodes have to be integers, ({i}, {j}) are not")

                basket[i] = basket.get(i, []) + [j]

        return basket

    def __init__(self, edges: List[Tuple] = [], n: int = 0):
        self.nodes = n
        self.edges = set(edges)
        self.edge_basket = self.get_edge_basket(edges)

    def from_edges(self, edges: List[Tuple]):
        max_node = 0

        for i, j in edges:
            if max_node - 1 < i:
                max_node = i + 1

            if max_node - 1 < j:
                max_node = j + 1

        self.nodes = max_node
        self.edges = set(edges)
        self.edge_basket = self.get_edge_basket(edges)


class WeightedNetwork(Network):
    def __init__(self, edges: List[Tuple] = [], n: int = 0):
        super().__init__(edges, n)

    def from_edges(self, edges: List[Tuple]):
        max_node = 0

        for i, j, _ in edges:
            if max_node - 1 < i:
                max_node = i + 1

            if max_node - 1 < j:
                max_node = j + 1

        self.nodes = max_node
        self.edges = set(edges)
        self.edge_basket = self.get_edge_basket(edges, weighted=True)
