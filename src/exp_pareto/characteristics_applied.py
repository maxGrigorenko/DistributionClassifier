from dataclasses import dataclass

import networkx as nx
import numpy as np

import networkx as nx

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
    max_independent_set = nx.algorithms.approximation.maximum_independent_set(graph)
    return len(max_independent_set)

def get_minimum_dominating_set_size(graph):
    min_dominating_set = nx.approximation.dominating_set(graph)
    return len(min_dominating_set)

def get_minimum_clique_cover_size(G): # мин число клик, которыми можно покрыть граф
    clique_cover = nx.approximation.min_weighted_clique_cover(G)
    return len(clique_cover)

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
    minimum_clique_cover_size: int

def create_characteristics_single(graph) -> CharacteristicsSingle:
    return CharacteristicsSingle(
        max_degree=get_max_degree(graph),
        min_degree=get_min_degree(graph),
        components=get_components(graph),
        number_of_articulation_points=get_number_of_articulation_points(graph),
        number_of_triangles=get_number_of_triangles(graph),
        chromatic=get_chromatic(graph),
        max_independent_set_size=get_max_independent_set_size(graph),
        minimum_dominating_set_size=get_minimum_dominating_set_size(graph),
        minimum_clique_cover_size=get_minimum_clique_cover_size(graph)
    )
