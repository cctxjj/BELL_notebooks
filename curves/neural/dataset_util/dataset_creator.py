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
        n: int=1000):
    """
    Creates a dataset of n parameters e [0, 1] for unsupervised training.
    """
    return tf.data.Dataset.from_tensor_slices(np.random.rand(n, 1).astype(np.float32))
