import numpy as np
import tensorflow as tf

from curves.funtions import bernstein_polynomial
from util.datasets.dataset_creator import create_random_bez_points


def drag_curve_squared_loss(n, t, y_pred):
    target_bez_vals = tf.constant(np.array([bernstein_polynomial(i, n, t) for i in range(n + 1)]).reshape(1, -1), dtype=tf.float32)
    deviation = tf.reduce_mean(tf.square(tf.subtract(y_pred, target_bez_vals))) # Formel 1 --> Abweichung von Bezierkurve
    y_pred_np = y_pred.numpy()



    return tf.Tensor([deviation], dtype=tf.float32)