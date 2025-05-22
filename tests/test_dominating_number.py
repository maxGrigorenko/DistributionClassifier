import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/normal_laplace'))

from graph_common_functions import Graph, distance_graph_constructor

class TestDominatingNumber(unittest.TestCase):
    def setUp(self):
        self.d = 2.0

    def test_empty_graph(self):
        arr = []
        graph = distance_graph_constructor(arr, self.d)
        self.assertEqual(graph.compute_dominating_number(self.d), 0)

    def test_single_node(self):
        arr = [5]
        graph = distance_graph_constructor(arr, self.d)
        self.assertEqual(graph.compute_dominating_number(self.d), 1)

    def test_all_isolated(self):
        arr = [1, 3, 5, 7]
        graph = distance_graph_constructor(arr, self.d)
        self.assertEqual(graph.compute_dominating_number(self.d), 4)

    def test_fully_connected(self):
        arr = [1, 2, 3, 4]
        graph = distance_graph_constructor(arr, d=5.0)
        self.assertEqual(graph.compute_dominating_number(d=5.0), 1)

    def test_optimal_case(self):
        arr = [1, 2, 4, 5, 7, 8]
        d = 3.0
        graph = distance_graph_constructor(arr, d)
        self.assertEqual(graph.compute_dominating_number(d), 2)

if __name__ == '__main__':
    unittest.main()
