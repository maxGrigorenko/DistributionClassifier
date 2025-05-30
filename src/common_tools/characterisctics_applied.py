from dataclasses import dataclass

import networkx as nx
from networkx.algorithms.dominating import dominating_set

from graphs import *

def get_max_degree(graph):
    return max(dict(graph.degree()).values())

def get_min_degree(graph):
    return min(dict(graph.degree()).values())

def get_mean_degree(graph):
    return (2 * graph.number_of_edges()) / graph.number_of_nodes()

def get_components(graph):
    return nx.number_connected_components(graph)


def get_number_of_articulation_points(graph):
    articulation_points = list(nx.articulation_points(graph))
    return len(articulation_points)


def get_number_of_triangles(graph):
    return sum(nx.triangles(graph).values()) // 3


def get_chromatic(graph):
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
    return max(coloring.values()) + 1


def get_max_independent_set_size(graph):
    inverted_graph = nx.DiGraph()
    for u, v in graph.edges():
        inverted_graph.add_edge(v, u)
    k = get_chromatic(inverted_graph)  # chromatic ~ max clique size
    return k


def get_minimum_dominating_set_size(graph, n_trials=5):
    res = float("inf")
    for _ in range(n_trials):
        res = min(len(dominating_set(graph)), res)
    return res

def get_minimum_dominating_set_size_for_dist(graph: Distance_Graph):
    d = graph.d_distance

    sorted_numbers = list(graph.get_numbers())
    sorted_numbers.sort()

    i = 0
    n = graph.n_vertexes

    count = 0
    while i < n:
        current_value = sorted_numbers[i]
        max_reach = current_value + d

        # Находим последнюю вершину в интервале [current_value, max_reach)
        j = i
        while j < n and sorted_numbers[j] < max_reach:
            j += 1

        # Выбираем последнюю вершину интервала
        selected = sorted_numbers[j - 1]
        count += 1

        # Пропускаем все вершины, покрытые выбранной
        i = j
        while i < n and sorted_numbers[i] < selected + d:
            i += 1

    return count


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
        minimum_dominating_set_size=get_minimum_dominating_set_size(graph),
    )


@dataclass
class ImplortantCharacteristicsSingle:
    components: int
    number_of_triangles: int
    chromatic: int
    minimum_dominating_set_size: int


def create_important_characteristics_single(graph) -> ImplortantCharacteristicsSingle:
    return ImplortantCharacteristicsSingle(
        components=get_components(graph),
        number_of_triangles=get_number_of_triangles(graph),
        chromatic=get_chromatic(graph),
        minimum_dominating_set_size=get_minimum_dominating_set_size(graph),
    )
