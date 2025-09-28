import math
import random

import numpy as np
import tensorflow as tf

from curves.funtions import bernstein_polynomial


def make_bernstein_dataset(
        num_samples: int=1000,
        degree: int=4):
    """
    Creates a dataset of points t e [0, 1] and the corresponding value of the Bernstein function for the given degree.
    """
    x = []
    y = []

    for _ in range(num_samples):
        n = degree
        t = np.random.rand()

        polynomial_vals = []
        for i in range(n + 1):
            polynomial_vals.append([bernstein_polynomial(i, n, t)])
        x.append([t])
        y.append([polynomial_vals])

    return tf.data.Dataset.from_tensor_slices(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32))

def create_n_parameter_values(
        n: int=1000,
        border_bottom: float=0.0,
        border_top: float=1.0):
    """
    Creates a dataset of n parameters e [border_bottom, border_top] for unsupervised training.
    """
    return tf.data.Dataset.from_tensor_slices(np.random.uniform(low=border_bottom, high=border_top, size=(n, 1)).astype(np.float32))

def create_random_bez_points(num_cont_points: int, x_min, x_max, y_min, y_max):
    last_x = x_min
    interval_size = math.floor((x_max - x_min)/num_cont_points)
    control_points = []
    for i in range(num_cont_points-2):
        control_points.append((last_x + random.randint(last_x, last_x+interval_size), random.randint(y_min, y_max)))
    return [(0, 0), *control_points, (x_max, 0)]

