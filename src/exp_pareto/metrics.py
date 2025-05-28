from typing import Union

import numpy as np
import pandas as pd


def calc_I_error(
    A: Union[np.ndarray, pd.DataFrame], first_points: Union[np.ndarray, pd.DataFrame]
):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(first_points, pd.DataFrame):
        first_points = first_points.to_numpy()

    is_in_A = np.any(
        np.all(first_points[:, None] == A, axis=2), axis=1
    )  # точки из first_points, не попавшие в A,
    # определяют ошибку I рода
    missed_points_count = np.sum(~is_in_A)
    return missed_points_count / first_points.shape[0]


def calc_power(
    A: Union[np.ndarray, pd.DataFrame], second_points: Union[np.ndarray, pd.DataFrame]
):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(second_points, pd.DataFrame):
        second_points = second_points.to_numpy()

    is_in_A = np.any(
        np.all(second_points[:, None] == A, axis=2), axis=1
    )  # точки из second_points, не попавшие в A
    # определяют мощность
    detected_points_count = np.sum(~is_in_A)
    return detected_points_count / second_points.shape[0]


def calc_I_error_wrapper(A_wrapper, first_points):
    missed_points_count = 0
    for point in first_points.values:
        if not A_wrapper.is_inside(point):
            missed_points_count += 1

    return missed_points_count / first_points.shape[0]


def calc_power_wrapper(A_wrapper, second_points):
    detected_points_count = 0
    for point in second_points.values:
        if not A_wrapper.is_inside(point):
            detected_points_count += 1

    return detected_points_count / second_points.shape[0]


def calc_I_error_clfr(classifier, first_points):
    return classifier.predict(first_points).sum() / first_points.shape[0]


def calc_power_clfr(classifier, second_points):
    return (classifier.predict(second_points).sum()) / second_points.shape[0]
