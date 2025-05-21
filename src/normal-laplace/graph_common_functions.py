import itertools
import time
import random


class Point:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class Graph:
    def __init__(self, graph_dict):
        self.graph = graph_dict  # {Point: set(Points}
        self.vertices = list(graph_dict.keys())

    def get_sorted_vertices(self):
        return sorted(self.vertices, key=lambda p: p.value)

    def compute_delta(self):
        delta = len(self.vertices)
        for v in self.vertices:
            delta = min(delta, len(self.graph[v]))
        return delta

    def compute_max_degree(self):
        max_degree = 0
        for v in self.vertices:
            max_degree = max(max_degree, len(self.graph[v]))
        return max_degree

    def compute_mean_degree(self):
        s = 0
        for v in self.vertices:
            s += len(self.graph[v])
        return s / len(self.vertices)

    def find_connected_components(self):
        visited = set()
        components = []
        for v in self.vertices:
            if v not in visited:
                component = []
                stack = [v]
                visited.add(v)
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for neighbor in self.graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                components.append(component)
        return components

    def compute_dominating_number(self, d):
        if not self.vertices:
            return 0

        sorted_vertices = self.get_sorted_vertices()
        count = 0
        n = len(sorted_vertices)
        i = 0

        while i < n:
            current_value = sorted_vertices[i].value
            max_reach = current_value + d

            # Находим последнюю вершину в интервале [current_value, max_reach)
            j = i
            while j < n and sorted_vertices[j].value < max_reach:
                j += 1

            # Выбираем последнюю вершину интервала
            selected = sorted_vertices[j - 1]
            count += 1

            # Пропускаем все вершины, покрытые выбранной
            i = j
            while i < n and sorted_vertices[i].value < selected.value + d:
                i += 1

        return count


def knn_graph_constructor(arr, k):
    points = [Point(name=i, value=val) for i, val in enumerate(arr)]
    graph_dict = {}

    for p in points:
        sorted_points = sorted(
            [other for other in points if other != p],
            key=lambda x: abs(x.value - p.value),
        )
        neighbors = set(sorted_points[:k])
        graph_dict[p] = neighbors

    return Graph(graph_dict)


def distance_graph_constructor(arr, d):
    points = [Point(name=i, value=val) for i, val in enumerate(arr)]
    graph_dict = {}

    for p in points:
        neighbors = set()
        for other in points:
            if p != other and abs(other.value - p.value) < d:
                neighbors.add(other)
        graph_dict[p] = neighbors

    return Graph(graph_dict)


# Тестовые примеры
if __name__ == "__main__":

    arr = [i / 3.14 for i in range(10)]
    print(knn_graph_constructor(arr, 3).graph)

    arr = [i for i in range(10)]
    print(distance_graph_constructor(arr, 4.5).graph)
