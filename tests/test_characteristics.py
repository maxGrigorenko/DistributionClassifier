import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/common_tools/')))

from graphs import Distance_Graph, KNN_Graph
from characterisctics_applied import *

class TestDominatingNumber(unittest.TestCase):
    def setUp(self):
        self.d = 2.0

    def test_single_node(self):
        arr = np.array([5])
        graph = Distance_Graph(n=1, d_distance=self.d)
        graph.build_from_numbers(arr)

        self.assertEqual(get_minimum_dominating_set_size_for_dist(graph), 1)

    def test_all_isolated(self):
        arr = np.array([1, 3, 5, 7])
        graph = Distance_Graph(n=4, d_distance=1.0)
        graph.build_from_numbers(arr)

        self.assertEqual(get_minimum_dominating_set_size_for_dist(graph), 4)

    def test_fully_connected(self):
        arr = np.array([1, 2, 3, 4])
        graph = Distance_Graph(n=4, d_distance=5.0)
        graph.build_from_numbers(arr)

        self.assertEqual(get_minimum_dominating_set_size_for_dist(graph), 1)

    def test_optimal_case(self):
        arr = np.array([1, 2, 4, 5, 7, 8])
        graph = Distance_Graph(n=6, d_distance=3.0)
        graph.build_from_numbers(arr)

        self.assertEqual(get_minimum_dominating_set_size_for_dist(graph), 2)

class TestCliqueNumber(unittest.TestCase):
    def setUp(self):
        self.d = 2.0

    def test_single_node(self):
        arr = np.array([5])
        graph = Distance_Graph(n=1, d_distance=self.d)
        graph.build_from_numbers(arr)

        self.assertEqual(get_chromatic(graph), 1)

    def test_all_isolated(self):
        arr = np.array([1, 3, 5, 7])  # Все точки на расстоянии 2+ друг от друга
        graph = Distance_Graph(n=4, d_distance=1.5)
        graph.build_from_numbers(arr)

        self.assertEqual(get_chromatic(graph), 1)

    def test_fully_connected(self):
        arr = np.array([1, 2, 3, 4])  # Все точки в пределах d=2
        graph = Distance_Graph(n=4, d_distance=3.5)
        graph.build_from_numbers(arr)
        
        self.assertEqual(get_chromatic(graph), 4)

    def test_optimal_case(self):
        arr = np.array([1.0, 1.9, 2.0, 2.1, 3.0])
        graph = Distance_Graph(n=5, d_distance=1.0)
        graph.build_from_numbers(arr)

        self.assertEqual(get_chromatic(graph), 3)  # [1.9, 2.0, 2.1]

class TestKNNGraphCharacteristics(unittest.TestCase):
    def setUp(self):
        self.k = 2  # Количество соседей
        self.n = 5  # Количество вершин

    def test_characteristics_single(self):
        arr = np.array([1, 2, 3, 4, 5])  # Все точки в пределах k=2 соседей
        graph = KNN_Graph(n=self.n, k_neighbours=self.k)
        graph.build_from_numbers(arr)

        characteristics = create_characteristics_single(graph)

        self.assertEqual(characteristics.max_degree, 2)  # Максимальная степень
        self.assertEqual(characteristics.components, 1)  # Количество компонент
        self.assertEqual(characteristics.number_of_articulation_points, 3)  # Артикулляционные точки
        self.assertEqual(characteristics.number_of_triangles, 0)  # Треугольники
        self.assertEqual(characteristics.chromatic, 2)  # Хроматическое число
        self.assertEqual(characteristics.max_independent_set_size, 2)  # Максимальный независимый набор
        self.assertEqual(characteristics.minimum_dominating_set_size, 3)  # Минимальный доминирующий набор

class TestDistanceGraphCharacteristics(unittest.TestCase):
    def setUp(self):
        self.d = 2.5

    def test_characteristics_single(self):
        arr = np.array([1, 2, 3, 5, 6])  # Все точки в пределах d=2.5
        graph = Distance_Graph(n=5, d_distance=self.d)
        graph.build_from_numbers(arr)

        characteristics = create_characteristics_single(graph)

        self.assertEqual(characteristics.max_degree, 3)  # Максимальная степень
        self.assertEqual(characteristics.min_degree, 1)  # Минимальная степень
        self.assertEqual(characteristics.components, 1)  # Количество компонент
        self.assertEqual(characteristics.number_of_articulation_points, 2)  # Артикулляционные точки
        self.assertEqual(characteristics.number_of_triangles, 1)  # Треугольники
        self.assertEqual(characteristics.chromatic, 3)  # Хроматическое число
        self.assertEqual(characteristics.max_independent_set_size, 2)  # Максимальный независимый набор
        self.assertEqual(characteristics.minimum_dominating_set_size, 2)  # Минимальный доминирующий набор

if __name__ == '__main__':
    unittest.main()
