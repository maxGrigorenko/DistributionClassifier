import matplotlib.pyplot as plt
import numpy as np

def draw_two_points_sets(first_points, second_points, title = "", xlabel = "", ylabel = "", first_legend = "Первое множество", second_legend = "Второе множество"):
    first_points = np.array(first_points)
    second_points = np.array(second_points)

    plt.figure(figsize=(10, 6))

    plt.scatter(first_points[:, 0], first_points[:, 1], color='blue', label=first_legend, alpha=0.5)

    # Скатерплот для парето распределения
    plt.scatter(second_points[:, 0], second_points[:, 1], color='orange', label=second_legend, alpha=0.5)

    # Настройки графика
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def draw_numbers_indexes_scatter(numbers, title = "", xlabel = "", ylabel = "", with_lines=False):
    indices = list(range(len(numbers)))

    # Создаем скатерплот
    plt.figure(figsize=(10, 6))
    plt.scatter(indices, numbers, color='blue', marker='o', s=1)
    if with_lines:
        plt.plot(indices, numbers, color='orange', linestyle='-', linewidth=1)

    # Добавляем заголовок и метки осей
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Отображаем график
    plt.grid()
    plt.show()