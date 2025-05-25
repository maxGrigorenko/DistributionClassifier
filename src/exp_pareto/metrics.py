import numpy as np
import pandas as pd
from typing import Union

def calc_I_error(A: Union[np.ndarray, pd.DataFrame], first_points: Union[np.ndarray, pd.DataFrame]):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(first_points, pd.DataFrame):
        first_points = first_points.to_numpy()
    
    is_in_A = np.any(
        np.all(first_points[:, None] == A, axis=2), axis=1
    )  # точки из first_points, не попавшие в A,
    # определяют ошибку I рода
    missed_points_count = np.sum(~is_in_A)
    return missed_points_count / first_points.size


def calc_power(A: Union[np.ndarray, pd.DataFrame], second_points: Union[np.ndarray, pd.DataFrame]):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(second_points, pd.DataFrame):
        second_points = second_points.to_numpy()
    
    is_in_A = np.any(
        np.all(second_points[:, None] == A, axis=2), axis=1
    )  # точки из second_points, не попавшие в A
    # определяют мощность
    detected_points_count = np.sum(~is_in_A)
    return detected_points_count / second_points.size
