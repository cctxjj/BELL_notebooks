import random

import numpy as np
import tensorflow as tf

from curves.funtions import bernstein_polynomial
from util.datasets.dataset_creator import create_random_curve_points


def drag_curve_squared_loss(
        model: tf.keras.Sequential,
        degree: int):
    cont_points = create_random_curve_points(6, random.randint(0, 3), random.randint(6, 13), 0,
                                                 random.randint(1, 15)) # TODO: check for effect of switching num_cont_points to random
    curve_points = []
    y_pred = model(inputs=tf.constant([[t/200] for t in range(200)], dtype=tf.float32))

    target_bez_vals = tf.constant(np.array([bernstein_polynomial(i, degree, t/200) for i in range(degree + 1)] for t in range(200)).reshape(1, -1), dtype=tf.float32)
    deviation = tf.reduce_mean(tf.square(tf.subtract(y_pred, target_bez_vals))) # Formel 1 --> Abweichung von Bezierkurve
    y_pred_np = y_pred.numpy()
    print(y_pred_np)

    return deviation