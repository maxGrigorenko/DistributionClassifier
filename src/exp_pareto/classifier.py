import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm
import sys
import os

from metrics import *
from visualisations import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../common_tools')))
from characteristics_applied import *
from graphs import Distance_Graph

class ConvexHullWrapper:
    def __init__(self, points_df):
        self.points = points_df.values
        self.hull = ConvexHull(self.points)
        self.delaunay = Delaunay(self.points[self.hull.vertices])

    def is_inside(self, point):
        point = np.array(point)
        return self.delaunay.find_simplex(point) >= 0


class DistibutionClassifier:
    lambda_param = 2 / np.sqrt(3)
    alpha_param = 3
    d = 0.005

    def __init__(self, n, observations_count=50):
        self.n_param = n  # number of rand var's instances
        self.observations_count = observations_count
        self.fitted = False

        self.exp_points = None
        self.pareto_points = None

        self.unique_exp_points = None
        self.unique_pareto_points = None

        self.A = None
        self.A_wrapper = None

    def generate_chars_points(self):
        print("Generating characteristics...")
        self.exp_points = []
        self.pareto_points = []

        for _ in tqdm(range(self.observations_count)):
            numbers_exp = np.random.exponential(1 / self.lambda_param, self.n_param)
            numbers_pareto = np.random.pareto(self.alpha_param, self.n_param) + 1

            distance_graph_exp = Distance_Graph(n=self.n_param, d_distance=self.d)
            distance_graph_pareto = Distance_Graph(n=self.n_param, d_distance=self.d)
            distance_graph_exp.build_from_numbers(numbers_exp)
            distance_graph_pareto.build_from_numbers(numbers_pareto)

            chars_exp = create_characteristics_single(distance_graph_exp)
            chars_pareto = create_characteristics_single(distance_graph_pareto)

            self.exp_points.append(chars_exp)
            self.pareto_points.append(chars_pareto)

        self.exp_points = pd.DataFrame(
            [vars(characteristic) for characteristic in self.exp_points]
        )
        self.pareto_points = pd.DataFrame(
            [vars(characteristic) for characteristic in self.pareto_points]
        )

        self.unique_exp_points = self.exp_points.drop_duplicates()
        self.unique_pareto_points = self.pareto_points.drop_duplicates()

        print("\nCharacteristics generated!")

    def generate_important_chars_points(self):
        print("Generating characteristics...")
        self.exp_points = []
        self.pareto_points = []

        for _ in tqdm(range(self.observations_count)):
            numbers_exp = np.random.exponential(1 / self.lambda_param, self.n_param)
            numbers_pareto = np.random.pareto(self.alpha_param, self.n_param) + 1

            distance_graph_exp = Distance_Graph(n=self.n_param, d_distance=self.d)
            distance_graph_pareto = Distance_Graph(n=self.n_param, d_distance=self.d)
            distance_graph_exp.build_from_numbers(numbers_exp)
            distance_graph_pareto.build_from_numbers(numbers_pareto)

            chars_exp = create_important_characteristics_single(distance_graph_exp)
            chars_pareto = create_important_characteristics_single(
                distance_graph_pareto
            )

            self.exp_points.append(chars_exp)
            self.pareto_points.append(chars_pareto)

        self.exp_points = pd.DataFrame(
            [vars(characteristic) for characteristic in self.exp_points]
        )
        self.pareto_points = pd.DataFrame(
            [vars(characteristic) for characteristic in self.pareto_points]
        )

        self.unique_exp_points = self.exp_points.drop_duplicates()
        self.unique_pareto_points = self.pareto_points.drop_duplicates()

        print("\nCharacteristics generated!")

    def get_points_dataset(self):
        join_column_name = "distribution_type"
        first_points, second_points = self.exp_points.copy(), self.pareto_points.copy()

        first_points[join_column_name] = 0
        second_points[join_column_name] = 1
        all_points = pd.concat([first_points, second_points], ignore_index=True)
        all_points.iloc[[0, -1]]
        X, y = all_points.drop(join_column_name, axis=1), all_points[[join_column_name]]

        return X, y

    def fit(self, exp_points_for_test, pareto_points_for_test, verbose=False):
        print("Building A...")

        self.A = self.unique_exp_points.copy()

        I_errors = []
        powers = []
        for i in tqdm(range(self.A.shape[0])):
            points_powers = {}

            for exp_point_to_remove in self.A.values:
                A_new = self.A.drop(
                    index=self.A[(self.A == exp_point_to_remove).all(axis=1)].index
                )
                A_new_wrapper = ConvexHullWrapper(A_new)

                # calc I error
                I_error = calc_I_error_wrapper(A_new_wrapper, exp_points_for_test)

                # I_errors.append(I_error)

                if I_error >= 0.05:
                    continue

                # lucky. calc and save power
                power = calc_power_wrapper(A_new_wrapper, pareto_points_for_test)

                points_powers[tuple(exp_point_to_remove)] = power

                # powers.append(power)

            if len(points_powers) == 0:
                break

            best_point_to_remove = np.array(max(points_powers, key=points_powers.get))
            self.A = self.A.drop(
                index=self.A[(self.A == best_point_to_remove).all(axis=1)].index
            )
            self.A_wrapper = ConvexHullWrapper(self.A)

            I_errors.append(calc_I_error_wrapper(self.A_wrapper, exp_points_for_test))
            powers.append(points_powers[tuple(exp_point_to_remove)])

        self.A_wrapper = ConvexHullWrapper(self.A)
        power = calc_power_wrapper(self.A_wrapper, pareto_points_for_test)
        I_error = calc_I_error_wrapper(self.A_wrapper, exp_points_for_test)

        print(f"A builded with power = {power}, I_error = {I_error}!")

        if verbose:
            self.draw_metrics(I_errors, powers)

    def predict_item(self, point: np.array):
        return not self.A_wrapper.is_inside(point)

    def predict_items(self, points: pd.DataFrame):
        results = []
        for index, point in points.iterrows():
            result = self.predict_item(point.values)
            results.append(result)
        return np.array(results)

    def draw_metrics(self, I_errors, powers):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].scatter(range(len(I_errors)), I_errors, color="blue", alpha=0.6, s=10)
        axs[0].set_title("Ошибка I рода по итерациям")
        axs[0].set_xlabel("Итерация")
        axs[0].set_ylabel("Ошибка I рода")
        axs[0].grid()

        axs[1].scatter(range(len(powers)), powers, color="orange", alpha=0.6, s=10)
        axs[1].set_title("Мощность по итерациям")
        axs[1].set_xlabel("Итерация")
        axs[1].set_ylabel("Мощность")
        axs[1].grid()

        plt.tight_layout()
        plt.show()


# clfr = DistibutionClassifier(n=50, observations_count=500)
# clfr_tester = DistibutionClassifier(n=50, observations_count=100)

# clfr.generate_important_chars_points()
# clfr_tester.generate_important_chars_points()

# clfr.fit(clfr_tester.exp_points, clfr_tester.pareto_points)
