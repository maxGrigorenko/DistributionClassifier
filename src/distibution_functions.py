import numpy as np
from graph_common_functions import *

def generate_normal(sigma, n):
    return np.random.normal(0, sigma, n)

def generate_laplace(beta, n):
    return np.random.laplace(0, beta, n)

def avg_chars(number_of_experiments, n, graph_type, distribution, sigma=0, beta=0, k=0, d=0.0):
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
            g = knn_graph_constructor(array, k)
            delta = g.compute_delta()
            characteristic_arr.append(delta)

        elif graph_type == "distance":
            g = distance_graph_constructor(array, d)
            dominating_number = g.compute_dominating_number(d=d)
            characteristic_arr.append(dominating_number)

        else:
            print("Incorrect graph_type")
            exit(1)

    return np.mean(characteristic_arr)
