import matplotlib.pyplot as plt
import numpy as np


def draw_two_points_sets(
    first_points,
    second_points,
    title="",
    xlabel="",
    ylabel="",
    first_legend="Первое множество",
    second_legend="Второе множество",
):
    first_points = np.array(first_points)
    second_points = np.array(second_points)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        first_points[:, 0],
        first_points[:, 1],
        color="blue",
        label=first_legend,
        alpha=0.5,
    )

    # Скатерплот для парето распределения
    plt.scatter(
        second_points[:, 0],
        second_points[:, 1],
        color="orange",
        label=second_legend,
        alpha=0.5,
    )

    # Настройки графика
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()


def draw_numbers_indexes_scatter(
    numbers, title="", xlabel="", ylabel="", with_lines=False
):
    indices = list(range(len(numbers)))

    # Создаем скатерплот
    plt.figure(figsize=(10, 6))
    plt.scatter(indices, numbers, color="blue", marker="o", s=1)
    if with_lines:
        plt.plot(indices, numbers, color="orange", linestyle="-", linewidth=1)

    # Добавляем заголовок и метки осей
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Отображаем график
    plt.grid()
    plt.show()


def draw_sorted_barplot(
    numbers, names, title="Важности характеристик", xlabel="Признаки", ylabel="Важность"
):
    indices = np.argsort(numbers)[::-1]
    sorted_numbers = numbers[indices]
    sorted_feature_names = names[indices]

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(sorted_numbers)), sorted_numbers, align="center")
    plt.xticks(range(len(sorted_numbers)), sorted_feature_names, rotation=60)
    plt.xlim([-1, len(sorted_numbers)])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def draw_two_sets_scatter_plots_matrix(
    first_points,
    second_points,
    title,
    first_label="exp_points",
    second_label="pareto_points",
):
    numeric_columns = first_points.select_dtypes(include="number").columns
    num_columns = len(numeric_columns)
    fig, axes = plt.subplots(num_columns, num_columns, figsize=(20, 20))

    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns):
            if i == j:
                axes[i, j].text(0.5, 0.5, col1, fontsize=12, ha="center", va="center")
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            else:
                axes[i, j].scatter(
                    first_points[col1],
                    first_points[col2],
                    color="blue",
                    label=first_label,
                    alpha=0.5,
                )
                axes[i, j].scatter(
                    second_points[col1],
                    second_points[col2],
                    color="red",
                    label=second_label,
                    alpha=0.5,
                )

                axes[i, j].set_title(" ")
                axes[i, j].set_xlabel(col1)
                axes[i, j].set_ylabel(col2)
                axes[i, j].legend()
                axes[i, j].grid()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
