import math

import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import SymbolicTensor
from tensorflow.python.keras.losses import MeanSquaredError

import util.visualisations as vis

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

def make_bernstein_dataset(num_samples=1000, degree=4):
    """
    Erzeugt Trainingsdaten für ein KNN:
    Input = [i, t]
    Target = B_{i,n}(t)
    """
    x = []
    y = []

    for _ in range(num_samples):
        n = degree
        t = np.random.rand()

        # Input-Feature: (n, i, t)
        polynomial_vals = []
        for i in range(n + 1):
            polynomial_vals.append([bernstein_polynomial(i, n, t)])
        x.append([t])
        y.append([polynomial_vals])

    x = np.array(x, dtype=np.float32)  # Shape (samples, 3)
    y = np.array(y, dtype=np.float32)  # Shape (samples, 1)
    return x, y
    # (return x, y)

# 2 inputs: parameter, control_point i
degree = 4

model = tf.keras.Sequential(
    layers = [

        tf.keras.layers.InputLayer(shape = (1,)),
        tf.keras.layers.Dense(32, activation = "relu"),
        tf.keras.layers.Dense(32, activation = "relu"),
        tf.keras.layers.Dense(degree + 1, activation = "softmax")
        ]
)

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
model.compile(optimizer = optimizer, loss = "mse")


# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

#model.fit(train_dataset, epochs=5)

def train_step(data: tf.Tensor):
    with tf.GradientTape() as tape:
        y_pred = model(data, training=True)
        n = degree  # nur Beispiel: n auslesen
        t = float(x.numpy()[0])
        target = np.array([bernstein_polynomial(i, n, t) for i in range(n + 1)],
                          dtype=np.float32).reshape(1, -1)
        target = tf.constant(target, dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(tf.subtract(y_pred, target)))

    # Backprop
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

train_dataset = tf.data.Dataset.from_tensor_slices(make_bernstein_dataset(num_samples=1000, degree=degree))

for epoch in range(10):
    loss = -1
    ind = 1
    for x, y in train_dataset:
        print(f"Datasample {ind}")
        loss = train_step(x)
        ind += 1
    print(f"Epoch {epoch + 1}: Loss = {loss.numpy()}")

control_points = [(1, 2), (2, 4), (3, 2), (5, 1), (6, -2)]

curve_points = []

'''

--> Output to ensure numerical correctness 

pol_vals = model(inputs=tf.constant([[0.5]], dtype=tf.float32)).numpy()
correct_vals = []
for i in range(degree + 1):
    correct_vals.append(bernstein_polynomial(i, degree, 0.5))

for index in range(len(pol_vals[0])):
    print(f"correct: {correct_vals[index]}; calculated: {pol_vals[0][index]}")
'''

for t in range(1000):
    cur_x = 0
    cur_y = 0
    pol_vals = model(inputs=tf.constant([[t / 1000]], dtype=tf.float32)).numpy()
    for i in range(degree + 1):
        bernstein_polynomial_value = pol_vals[0][i]
        cur_x += bernstein_polynomial_value * control_points[i][0]
        cur_y += bernstein_polynomial_value * control_points[i][1]
    curve_points.append((cur_x, cur_y))

vis.visualize_curve(curve_points, control_points, True)






