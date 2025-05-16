import numpy as np
from tqdm import tqdm

from characteristics import *
from graphs import Distance_Graph
from metrics import *
from visualisations import *


class DistibutionClassifierSimplified:
    lambda_param = 2 / np.sqrt(3)
    alpha_param = 3
    k = 15
    d = 0.005

    def __init__(self, n, A_canditates_count=10):
        self.n_param = n
        self.A_canditates_count = A_canditates_count
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

        for _ in tqdm(range(self.n_param)):
            numbers_exp = np.random.exponential(1 / self.lambda_param, self.n_param)
            numbers_pareto = np.random.pareto(self.alpha_param, self.n_param) + 1

            distance_graph_exp = Distance_Graph(n=self.n_param, d_distance=self.d)
            distance_graph_pareto = Distance_Graph(n=self.n_param, d_distance=self.d)
            distance_graph_exp.build_from_numbers(numbers_exp)
            distance_graph_pareto.build_from_numbers(numbers_pareto)

            chars = create_characteristics(
                None, distance_graph_exp, None, distance_graph_pareto, "dist"
            )

            self.exp_points.append(
                (chars.distance_exp_components, chars.distance_pareto_chromatic)
            )
            self.pareto_points.append(
                (chars.distance_pareto_components, chars.distance_pareto_chromatic)
            )

        self.exp_points = np.array(self.exp_points)
        self.pareto_points = np.array(self.pareto_points)

        self.unique_exp_points = np.unique(self.exp_points, axis=0)
        self.unique_pareto_points = np.unique(self.pareto_points, axis=0)

        print("Characteristics generated!")

    def build_A_candidate(self, candidate_index=0):
        print("\tBuinding A_candidate_...", candidate_index, sep="")

        A_candidate = self.unique_exp_points.copy()
        np.random.shuffle(A_candidate)

        I_errors = []
        powers = []
        for i in tqdm(range(A_candidate.size)):
            points_powers = {}

            for exp_point_to_remove in A_candidate:
                A_new = A_candidate[~np.all(A_candidate == exp_point_to_remove, axis=1)]

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
                continue

            best_point_to_remove = max(points_powers, key=points_powers.get)
            A_candidate = A_candidate[
                ~np.all(A_candidate == best_point_to_remove, axis=1)
            ]

        power = calc_power(A_candidate, self.pareto_points)
        print(f"\tA_candidate{candidate_index} builded with power = {power}!")

        return A_candidate, power

    def build_A(self):
        print("Building A...")

        A_candidates, powers = [], []
        for i in tqdm(self.A_canditates_count):
            A_candidate, power = self.build_A_candidate(i)
            A_candidates.append(A_candidate)
            powers.append(power)

        A_candidates = list(enumerate(A_candidates))
        self.A, power = max(A_candidates, key=lambda item: powers[item[1]])

        print(f"A builded with power={power}!")

    def fit(self):
        self.generate_chars_points()

        self.build_A()

    def draw_A(self):
        draw_two_points_sets(
            self.A,
            self.unique_pareto_points,
            title="Скатерплоты для A и уникальных парето точек",
            xlabel="Число компонент",
            ylabel="Хроматическое число",
            first_legend="A",
            second_legend="Парето точки",
        )

    def predict(self, point):
        return point in self.A
