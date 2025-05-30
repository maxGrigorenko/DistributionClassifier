import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


class KNN_Graph(nx.Graph):
    n_vertexes: int
    k_neighbours: int
    numbers: np.ndarray

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.n_vertexes = attr["n"]
        self.k_neighbours = attr["k_neighbours"]
        self.numbers = None

    def build_from_numbers(self, numbers: np.ndarray):
        assert numbers.size == self.n_vertexes

        numbers_reshaped = numbers.reshape(-1, 1)
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbours).fit(numbers_reshaped)
        distances, indices = nbrs.kneighbors(numbers_reshaped)

        for i in range(self.n_vertexes):
            self.add_node(i, pos=(numbers_reshaped[i][0], numbers_reshaped[i][0]))

        for i in range(self.n_vertexes):
            for j in indices[i][1:]:
                self.add_edge(i, j)

        self.numbers = numbers.copy()

    def draw(self):
        plt.figure(figsize=(8, 6))
        nx.draw(
            self,
            nx.spring_layout(self),
            with_labels=True,
            node_size=50,
            node_color="lightblue",
            font_size=7,
            font_color="black",
            font_weight="bold",
        )
        plt.title("Изображение KNN-графа")
        plt.show()


class Distance_Graph(nx.Graph):
    n_vertexes: int
    d_distance: float
    numbers: np.ndarray

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.n_vertexes = attr["n"]
        self.d_distance = attr["d_distance"]
        self.numbers = None

    def build_from_numbers(self, numbers: np.ndarray):
        assert numbers.size == self.n_vertexes

        numbers_reshaped = numbers.reshape(-1, 1)
        distances = pairwise_distances(numbers_reshaped)
        for i in range(self.n_vertexes):
            self.add_node(i, label=numbers_reshaped[i][0])

        for i in range(self.n_vertexes):
            for j in range(i + 1, self.n_vertexes):
                if distances[i][j] < self.d_distance:
                    self.add_edge(i, j)

        self.numbers = numbers.copy()

    def get_numbers(self):
        return self.numbers.copy()

    def draw(self):
        pos = nx.spring_layout(self)

        plt.figure(figsize=(8, 6))
        nx.draw(
            self,
            pos,
            with_labels=True,
            node_size=50,
            node_color="lightblue",
            font_size=7,
            font_color="black",
            font_weight="bold",
        )
        plt.title("Изображение дистанционного графа")
        plt.show()
