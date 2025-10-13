import os
import random
import sys

import vis

import numpy as np
import tensorflow as tf

from curves.func_based.bézier_curve import bezier_curve
from curves.funtions import bernstein_polynomial
import util.graphics.visualisations as vis
from curves.neural.custom_metrics.drag_evaluation import DragEvaluator


def make_bernstein_dataset(
        num_samples: int = 1000,
        degree: int = 4):
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
        n: int = 1000,
        border_bottom: float = 0.0,
        border_top: float = 1.0):
    """
    Creates a dataset of n parameters element of [border_bottom, border_top] for unsupervised training.
    """
    return tf.data.Dataset.from_tensor_slices(
        np.random.uniform(low=border_bottom, high=border_top, size=(n, 1)).astype(np.float32))


def create_random_curve_points(
        num_cont_points: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float):
    last_x = x_min
    interval_size = (x_max - x_min) / num_cont_points
    control_points = []
    last_y = y_min
    for i in range(num_cont_points-2):
        if i > (num_cont_points - 2 / 2) or last_y >= y_max:
            new_y = np.random.uniform(y_min, last_y)
        else:
            new_y = np.random.uniform(last_y, last_y + y_max / interval_size)
        last_x += np.random.uniform(0, interval_size)
        control_points.append((last_x, new_y))
        last_y = new_y
    return [(x_min, 0), *control_points, (x_max, 0)]


def create_bez_curves_drag_coef_dataset(
        degree: int = 5,
        points_num: int = 100,
        samples: int = 5000 ,
        file_name: str = "bez_curves_cd"
):
    # Structure: (curve (represented as n points): list, alpha , c_d value)
    curves = []
    for i in range(samples):
        print(f"\rCreating curve {i+1}/{samples}", end="")
        sys.stdout.flush()
        cont_points = create_random_curve_points(degree, random.randint(0, 3), random.randint(6, 13), 0,
                                                 random.randint(1, 15))
        points = bezier_curve(cont_points, points_num)
        curves.append(points)
    # TODO: in wie Fern
    print("\nCurve creation done. Evaluating drag coefficients.\n")
    values = []
    i = 0
    curves_validated = []
    for curve in curves:
        i+=1
        print(f"\rEvaluating curve {i}/{samples} | valid datapoints collected: {len(values)}", end="")
        sys.stdout.flush()
        alpha, cd = DragEvaluator(curve, save_airfoil=False, range=16, start_angle=0, specification="dataset_creation_test_3").find_valid_alpha_cd()
        if alpha is not None:
            values.append((alpha, cd))
            curves_validated.append(curve)
    os.makedirs(os.path.dirname(f"C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/datasets/{file_name}.npz"),
                exist_ok=True)
    np.savez(f"C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/datasets/{file_name}.npz",
                     points=curves_validated, values=values)
    if len(curves_validated) != len(values):
        raise Exception("Mismatch between curves and values in dataset creation.")
    print("\nDataset created and saved under given file name.")
    return len(curves_validated), len(values)

# TODO: test correct alg for turning & adjusting curves

n1, n2 = create_bez_curves_drag_coef_dataset(file_name="bez_curves_cd_test")
print(n1, n2)

def __plot_n_example_random_bez_curves__(
        amount: int,
        points_num: int = 300):
    # TODO: comment
    for i in range(amount):
        cont_points = create_random_curve_points(random.randint(4, 8), random.randint(0, 3), random.randint(6, 13), 0,
                                                 random.randint(1, 15))
        points = bezier_curve(cont_points, points_num)
        vis.visualize_curve(points, cont_points, True, file_name=f"bez_curve_{i + 1}",
                            save_path="C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/neural_curves/random_bez_curves")
        DragEvaluator(points, save_airfoil=True, range=30, start_angle=0, specification=f"random_bez_curve")
