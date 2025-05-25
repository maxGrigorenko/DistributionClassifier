import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from dataclasses import dataclass
from tqdm import tqdm
from itertools import product

from graphs import KNN_Graph, Distance_Graph
from characteristics_applied import *
from visualisations import *
from metrics import *

class DistibutionClassifier:
    lambda_param = 2 / np.sqrt(3)
    alpha_param = 3
    k = 15
    d = 0.005

    def __init__(self, n, observations_count=50):
        self.n_param = n # number of rand var's instances
        self.observations_count = observations_count
        self.fitted = False

        self.exp_points = None
        self.pareto_points = None

        self.unique_exp_points = None
        self.unique_pareto_points = None

        self.A = None

    def generate_chars_points(self):
        print("Generating characteristics...")
        self.exp_points = []
        self.pareto_points = []

        for _ in tqdm(range(self.observations_count)):
            numbers_exp = np.random.exponential(1/self.lambda_param, self.n_param)
            numbers_pareto = (np.random.pareto(self.alpha_param, self.n_param) + 1)

            distance_graph_exp = Distance_Graph(n=self.n_param, d_distance=self.d)
            distance_graph_pareto = Distance_Graph(n=self.n_param, d_distance=self.d)
            distance_graph_exp.build_from_numbers(numbers_exp)
            distance_graph_pareto.build_from_numbers(numbers_pareto)

            chars_exp = create_characteristics_single(distance_graph_exp)
            chars_pareto = create_characteristics_single(distance_graph_pareto)

            self.exp_points.append(chars_exp)
            self.pareto_points.append(chars_pareto)

        self.exp_points = pd.DataFrame([vars(characteristic) for characteristic in self.exp_points])
        self.pareto_points = pd.DataFrame([vars(characteristic) for characteristic in self.pareto_points])

        self.unique_exp_points = self.exp_points.drop_duplicates()
        self.unique_pareto_points = self.pareto_points.drop_duplicates()

        print("Characteristics generated!")

    def build_A(self): # TODO: change in terms of pd.DataFrame
        print("Building A...")

        A = self.unique_exp_points.copy()

        I_errors = []
        powers = []
        for i in tqdm(range(A.size)):
        # for i in range(A.size):
            points_powers = {}

            for exp_point_to_remove in A.values:
                A_new = A.drop(index=A[(A == exp_point_to_remove).all(axis=1)].index)

                # calc I error
                I_error = calc_I_error(A_new, self.exp_points)

                I_errors.append(I_error)

                if I_error >= 0.05:
                    continue

                # lucky. calc and save power
                power = calc_power(A_new, self.pareto_points)

                points_powers[tuple(exp_point_to_remove)] = power

                powers.append(power)

            if len(points_powers) == 0:
                break

            best_point_to_remove = max(points_powers, key=points_powers.get)
            A = A.drop(index=A[(A == best_point_to_remove).all(axis=1)].index)

        self.A = A
        power = calc_power(A, self.pareto_points)

        print(f"A builded with power = {power}!")

    def fit(self):
        self.generate_chars_points()

        self.build_A()

    def draw_A(self):
        draw_two_points_sets(self.A, self.unique_pareto_points, title='Скатерплоты для A и уникальных парето точек',
                     xlabel='Число компонент', ylabel='Хроматическое число',
                     first_legend='A', second_legend='Парето точки')

    def predict(self, point):
        return point in self.A
