from dataclasses import dataclass

import networkx as nx
import numpy as np

from graphs import Distance_Graph, KNN_Graph


def get_components(graph):
    return nx.number_connected_components(graph)


def get_chromatic(graph):
    chromatic_number = nx.coloring.greedy_color(graph, strategy="largest_first")
    unique_colors = set(chromatic_number.values())
    return len(unique_colors)


@dataclass
class Characteristics:
    knn_exp_components: int
    distance_exp_chromatic: int
    knn_pareto_components: int
    distance_pareto_chromatic: int
    knn_exp_chromatic: int
    distance_exp_components: int
    knn_pareto_chromatic: int
    distance_pareto_components: int


def create_characteristics(
    knn_graph_exp,
    distance_graph_exp,
    knn_graph_pareto,
    distance_graph_pareto,
    distrib_type=None,
):
    if distrib_type == "knn":
        return Characteristics(
            knn_exp_components=get_components(knn_graph_exp),
            knn_pareto_components=get_components(knn_graph_pareto),
            knn_exp_chromatic=get_chromatic(knn_graph_exp),
            knn_pareto_chromatic=get_chromatic(knn_graph_pareto),
            distance_exp_chromatic=-1,
            distance_pareto_chromatic=-1,
            distance_exp_components=-1,
            distance_pareto_components=-1,
        )
    elif distrib_type == "dist":
        return Characteristics(
            knn_exp_components=-1,
            knn_pareto_components=-1,
            knn_exp_chromatic=-1,
            knn_pareto_chromatic=-1,
            distance_exp_chromatic=get_chromatic(distance_graph_exp),
            distance_pareto_chromatic=get_chromatic(distance_graph_pareto),
            distance_exp_components=get_components(distance_graph_exp),
            distance_pareto_components=get_components(distance_graph_pareto),
        )


def get_characteristics(lambda_param, alpha_param, n, k, d, distrib_type=None):
    numbers_exp = np.random.exponential(1 / lambda_param, n)
    numbers_pareto = np.random.pareto(alpha_param, n) + 1

    knn_graph_exp = None
    knn_graph_pareto = None
    distance_graph_exp = None
    distance_graph_pareto = None

    if distrib_type == "knn":
        knn_graph_exp = KNN_Graph(n=n, k_neighbours=k)
        knn_graph_pareto = KNN_Graph(n=n, k_neighbours=k)
        knn_graph_exp.build_from_numbers(numbers_exp)
        knn_graph_pareto.build_from_numbers(numbers_pareto)

    elif distrib_type == "dist":
        distance_graph_exp = Distance_Graph(n=n, d_distance=d)
        distance_graph_pareto = Distance_Graph(n=n, d_distance=d)
        distance_graph_exp.build_from_numbers(numbers_exp)
        distance_graph_pareto.build_from_numbers(numbers_pareto)

    return create_characteristics(
        knn_graph_exp, distance_graph_exp, knn_graph_pareto, distance_graph_pareto
    )


def get_average_characteristics(lambda_param, alpha_param, n, k, d, distrib_type=None):
    characteristics_list = []
    for trial in range(5):
        characteristics = get_characteristics(
            lambda_param, alpha_param, n, k, d, distrib_type
        )
        characteristics_list.append(characteristics)

    r = None
    if distrib_type == "knn":
        r = Characteristics(
            knn_exp_components=np.mean(
                [data.knn_exp_components for data in characteristics_list]
            ),
            knn_pareto_components=np.mean(
                [data.knn_pareto_components for data in characteristics_list]
            ),
            knn_exp_chromatic=np.mean(
                [data.knn_exp_chromatic for data in characteristics_list]
            ),
            knn_pareto_chromatic=np.mean(
                [data.knn_pareto_chromatic for data in characteristics_list]
            ),
            distance_exp_chromatic=-1,
            distance_pareto_chromatic=-1,
            distance_exp_components=-1,
            distance_pareto_components=-1,
        )
    elif distrib_type == "dist":
        r = Characteristics(
            knn_exp_components=-1,
            knn_pareto_components=-1,
            knn_exp_chromatic=-1,
            knn_pareto_chromatic=-1,
            distance_exp_chromatic=np.mean(
                [data.distance_exp_chromatic for data in characteristics_list]
            ),
            distance_pareto_chromatic=np.mean(
                [data.distance_pareto_chromatic for data in characteristics_list]
            ),
            distance_exp_components=np.mean(
                [data.distance_exp_components for data in characteristics_list]
            ),
            distance_pareto_components=np.mean(
                [data.distance_pareto_components for data in characteristics_list]
            ),
        )

    return r
