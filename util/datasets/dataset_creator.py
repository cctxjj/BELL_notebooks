import random

import vis

import curves
import numpy as np
import tensorflow as tf

from curves.func_based.bézier_curve import bezier_curve
from curves.funtions import bernstein_polynomial
import util.graphics.visualisations as vis
from curves.neural.custom_metrics.drag_evaluation import DragEvaluator


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
    Creates a dataset of n parameters element of [border_bottom, border_top] for unsupervised training.
    """
    return tf.data.Dataset.from_tensor_slices(np.random.uniform(low=border_bottom, high=border_top, size=(n, 1)).astype(np.float32))

def create_random_bez_points(
        num_cont_points: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float):
    last_x = x_min
    interval_size = (x_max - x_min)/num_cont_points
    control_points = []
    last_y = y_min
    for i in range(num_cont_points-2):
        if i > (num_cont_points-2/2) or last_y >= y_max:
            new_y = np.random.uniform(y_min, last_y)
        else:
            new_y = np.random.uniform(last_y, last_y+y_max/interval_size)
        last_x += np.random.uniform(0, interval_size)
        control_points.append((last_x, new_y))
        last_y = new_y
    return [(x_min, 0), *control_points, (x_max, 0)]

def __create_example_random_bez_curves__(
        amount: int,
        points_num: int=300):
    # TODO: comment
    for i in range(amount):
        cont_points = create_random_bez_points(random.randint(4, 8), random.randint(0, 3), random.randint(6, 13), 0, random.randint(1, 15))
        points = bezier_curve(cont_points, points_num)
        vis.visualize_curve(points, cont_points, True, file_name=f"bez_curve_{i+1}", save_path="C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/neural_curves/random_bez_curves")
        DragEvaluator(points, save_airfoil=True, range=30, start_angle=0, name_appendix=f"random_bez_curve")


