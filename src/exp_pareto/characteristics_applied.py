from dataclasses import dataclass

import networkx as nx
import numpy as np

from graphs import Distance_Graph
from networkx.algorithms.dominating import dominating_set

def get_max_degree(graph):
    return max(dict(graph.degree()).values())

def get_min_degree(graph):
    return min(dict(graph.degree()).values())

def get_components(graph):
    return nx.number_connected_components(graph)

def get_number_of_articulation_points(graph):
    articulation_points = list(nx.articulation_points(graph))
    return len(articulation_points)

def get_number_of_triangles(graph):
    return sum(nx.triangles(graph).values()) // 3

def get_chromatic(graph):
    coloring = nx.coloring.greedy_color(graph, strategy='largest_first')
    return max(coloring.values()) + 1

def get_max_independent_set_size(graph):
    inverted_graph = nx.DiGraph()
    for u, v in graph.edges():
        inverted_graph.add_edge(v, u)
    return get_chromatic(inverted_graph) # chromatic ~ max clique size

def get_minimum_dominating_set_size(graph, n_trials=5):
    res = float('inf')
    for _ in range(n_trials):
        res = min(
            len(dominating_set(graph)),
            res
        )
    return res

@dataclass
class CharacteristicsSingle:
    max_degree: int
    min_degree: int
    components: int
    number_of_articulation_points: int
    number_of_triangles: int
    chromatic: int
    max_independent_set_size: int
    minimum_dominating_set_size: int

def create_characteristics_single(graph) -> CharacteristicsSingle:
    return CharacteristicsSingle(
        max_degree=get_max_degree(graph),
        min_degree=get_min_degree(graph),
        components=get_components(graph),
        number_of_articulation_points=get_number_of_articulation_points(graph),
        number_of_triangles=get_number_of_triangles(graph),
        chromatic=get_chromatic(graph),
        max_independent_set_size=get_max_independent_set_size(graph),
        minimum_dominating_set_size=get_minimum_dominating_set_size(graph)
    )
