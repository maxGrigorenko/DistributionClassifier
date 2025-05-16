import numpy as np


def calc_I_error(A, first_points):
    is_in_A = np.any(
        np.all(first_points[:, None] == A, axis=2), axis=1
    )  # точки из first_points, не попавшие в A,
    # определяют ошибку I рода
    missed_points_count = np.sum(~is_in_A)
    return missed_points_count / first_points.size


def calc_power(A, second_points):
    is_in_A = np.any(
        np.all(second_points[:, None] == A, axis=2), axis=1
    )  # точки из second_points, не попавшие в A
    # определяют мощность
    detected_points_count = np.sum(~is_in_A)
    return detected_points_count / second_points.size
