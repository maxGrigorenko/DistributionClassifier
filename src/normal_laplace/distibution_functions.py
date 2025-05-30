import os
import sys

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../common_tools"))
)
from characterisctics_applied import *
from graphs import *


def generate_normal(sigma, n):
    return np.random.normal(0, sigma, n)


def generate_laplace(beta, n):
    return np.random.laplace(0, beta, n)


def avg_chars(
    number_of_experiments, n, graph_type, distribution, sigma=0, beta=0, k=0, d=0.0
):
    characteristic_arr = []
    for t in range(number_of_experiments):
        if distribution == "normal":
            array = generate_normal(sigma, n)

        elif distribution == "laplace":
            array = generate_laplace(beta, n)

        else:
            print("Incorrect distribution")
            exit(1)

        if graph_type == "knn":
            graph = KNN_Graph(n=n, k_neighbors=k)
            graph.build_from_numbers(array)

            delta = get_min_degree(graph)
            characteristic_arr.append(delta)

        elif graph_type == "distance":
            graph = Distance_Graph(n=n, d_distance=d)
            graph.build_from_numbers(array)

            dominating_number = get_minimum_dominating_set_size_for_dist(array)
            characteristic_arr.append(dominating_number)

        else:
            print("Incorrect graph_type")
            exit(1)

    return np.mean(characteristic_arr)
