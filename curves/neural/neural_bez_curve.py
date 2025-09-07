import math

import numpy as np
import tensorflow as tf
import curves.recreation.bézier_curve as bez_c

class TestModel(tf.keras.Sequential):
    def __init__(self, layers):
        super().__init__()
        for layer in layers:
            self.add(layer)



def bernstein_polynomial(i: int,
                         n: int,
                         t: float):
    """
    :param i: number of the bernstein polynomial
    :param n: degree of the bernstein polynomial
    :param t: given parameter, curvepoint (on interval [0, 1]) to be calculated at
    :return: Value of the Bernstein Polynomial at point t
    """
    binomial_coefficient = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
    return binomial_coefficient * (t ** i) * ((1 - t) ** (n - i))


