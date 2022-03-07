from typing import List, Tuple, Optional, Union


class Network:
    def __init__(self, edges: List[Tuple] = [], n: int = 0):
        self.nodes = n
        self.edges = edges

    def from_edges(self, edges: List[Typle]):
        pass


class WeightNetwork(Network):
    def __init__(self, edges: List[Tuple] = [], n: int = 0, weights: Union[int, List] = 1):
        super().__init__(edges, n)
        self.weights = weights

    def from_edges(self, edges: List[Typle]):
        pass
