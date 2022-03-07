from typing import Any, List, Tuple, Optional, Union, Dict


def is_int(number: Any) -> bool:
    try:
        int(number)
        return True
    except TypeError:
        return False


def is_float(number: Any) -> bool:
    try:
        float(number)
        return True
    except TypeError:
        return False


class Network:
    @staticmethod
    def get_edge_basket(edges: List[Tuple], weighted: bool = False) -> Dict:
        basket = dict()
        if weighted:
            for i, j, w in edges:
                basket[i] = basket.get(i, []).append((j, w))
        else:
            for i, j in edges:
                basket[i] = basket.get(i, []).append(j)

        return basket

    def __init__(self, edges: List[Tuple] = [], n: int = 0):
        self.nodes = n
        self.edges = set(edges)
        self.edge_basket = self.get_edge_basket(edges)

    def from_edges(self, edges: List[Typle]):
        max_node = 0

        for i, j in edges:
            if not is_int(i) or not is_int(j):
                raise ValueError(f"Nodes have to be integers, ({i}, {j}) are not")

            if max_node > i:
                max_node = i

            if max_node > j:
                max_node = j

        self.nodes = max_node
        self.edges = set(edges)
        self.edge_basket = self.get_edge_basket(edges)


class WeightNetwork(Network):
    def __init__(self, edges: List[Tuple] = [], n: int = 0):
        super().__init__(edges, n)

    def from_edges(self, edges: List[Typle]):
        max_node = 0

        for i, j, w in edges:
            if not is_int(i) or not is_int(j):
                raise ValueError(f"Nodes have to be integers, ({i}, {j}) are not")

            if not is_float(w):
                raise ValueError(f"Weight has to be float, {w} is not")

            if max_node > i:
                max_node = i

            if max_node > j:
                max_node = j

        self.nodes = max_node
        self.edges = set(edges)
        self.edge_basket = self.get_edge_basket(edges, weighted=True)
