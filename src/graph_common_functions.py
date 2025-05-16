import itertools
import time
import random

class Graph:
    def __init__(self, graph_dict):
        self.graph = graph_dict
        self.n = len(graph_dict)
        self.vertices = list(graph_dict.keys())

    def compute_delta(self):
        m = self.n
        for v in self.vertices:
            m = min(m, len(self.graph[v]))

        return m

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

    def exact_dominating_number(self, timeout=40):
        start_time = time.time()
        components = self.find_connected_components()
        total = 0

        for comp in components:
            if time.time() - start_time > timeout:
                return None  # Прерываем выполнение

            if len(comp) == 1:
                total += 1
                continue

            subgraph = {v: self.graph[v] & set(comp) for v in comp}
            min_size = len(comp)
            for k in range(1, len(comp) + 1):
                for subset in itertools.combinations(comp, k):
                    if time.time() - start_time > timeout:
                        return None  # Прерываем выполнение

                    dominated = set(subset)
                    for v in subset:
                        dominated.update(subgraph[v])
                    if len(dominated) == len(comp):
                        min_size = k
                        break
                if min_size == k:
                    break
            total += min_size
        return total

    def optimized_greedy_dominating_set(self, max_time=10):
        dominated = set()
        solution = []
        vertices = set(self.vertices)

        # Жадный алгоритм
        while vertices - dominated:
            best_v = max(
                (v for v in vertices - dominated),
                key=lambda x: len(self.graph[x] - dominated),
                default=None
            )
            if best_v is None:
                break
            solution.append(best_v)
            dominated.update(self.graph[best_v])
            dominated.add(best_v)

        # Локальный поиск
        start_time = time.time()
        best_solution = solution.copy()
        best_size = len(solution)

        while time.time() - start_time < max_time:
            # Пытаемся удалить случайную вершину
            if not best_solution:
                break
            candidate = random.choice(best_solution)
            temp_solution = [v for v in best_solution if v != candidate]
            if self._is_dominating(temp_solution):
                best_solution = temp_solution
                best_size -= 1
        return best_size

    def _is_dominating(self, solution):
        dominated = set(solution)
        for v in solution:
            dominated.update(self.graph[v])
        return dominated == set(self.vertices)

    def compute_dominating_number(self, exact_timeout=40, greedy_time=10, verbose=True):
        # Пытаемся найти точное решение
        start_time = time.time()
        exact_result = self.exact_dominating_number(exact_timeout)
        if exact_result is not None:
            if verbose:
                print(f"Точное решение найдено за {time.time() - start_time:.2f} сек")
            return exact_result

        if verbose:
            print("Точное решение не найдено. Используется жадный алгоритм...")
        return self.optimized_greedy_dominating_set(greedy_time)

class Point:
    def __init__(self, name, value):
        self.name = name
        self.value = value

def knn_graph_constructor(arr, k):
    graph_dict = dict()
    points = []
    for i in range(len(arr)):
        p = Point(name=i, value=arr[i])
        points.append(p)

    for i in range(len(points)):
        p = points[i]
        graph_dict.update({p.name: set(list(map(lambda x: x.name, sorted(points, key=lambda x: abs(x.value - p.value))))[1:(k+1)])})

    return Graph(graph_dict)


def distance_graph_constructor(arr, d):
    graph_dict = dict()
    points = []
    for i in range(len(arr)):
        p = Point(name=i, value=arr[i])
        points.append(p)

    for i in range(len(points)):
        p = points[i]
        graph_dict.update({p.name: set([point.value for point in points if 0 < abs(point.value - p.value) < d])})

    return Graph(graph_dict)




# Тестовые примеры
if __name__ == "__main__":

    arr = [i/3.14 for i in range(10)]
    print(knn_graph_constructor(arr, 3).graph)


    arr = [i for i in range(10)]
    print(distance_graph_constructor(arr, 4.5).graph)
