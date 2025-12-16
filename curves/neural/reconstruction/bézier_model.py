import sys

import tensorflow as tf
import numpy as np

import util.graphics.visualisations as vis
from curves.funtions import bernstein_polynomial
from util.datasets.dataset_creator import create_n_parameter_values


"""
Approximierung der Bernsteinpolynome (Bézierkurve) mittels NN, Integration eines custom training loops
"""

# setting up variables
degree = 4
epochs = 30
samples = 1000

# model setup
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
# custom training loop
def train_step(data: tf.Tensor):
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model(data, training=True)

        # Loss calculation
        n = degree
        t = float(data.numpy())
        target = np.array([bernstein_polynomial(i, n, t) for i in range(n + 1)],
                          dtype=np.float32).reshape(1, -1)
        target = tf.constant(target, dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(tf.subtract(y_pred, target)))

    # Backprop
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# dataset creation and training
train_dataset = create_n_parameter_values(n=samples)

for epoch in range(epochs):
    loss = -1
    ind = 0

    # iteration over dataset, training (not using batches)
    for x in train_dataset:

        loss = train_step(x)
        ind += 1

        # output progress
        print(f"\rEpoche: {(epoch+1)}/{epochs} | Sample: {ind}/{samples} | Loss: {loss}", end="")
        if ind == samples:
            print(f"\rEpoch: {(epoch+1)}/{epochs} | Sample: {ind}/{samples} | Loss: {loss} | epoche completed")
        sys.stdout.flush()


# Test: creating an example curve
control_points = [(1, 2), (2, 4), (3, 2), (5, 1), (6, -2)]

curve_points = []

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

# TODO: Frage: Wo liegt Effizienz im Ansatz der Wichtung mit KNNs im Vgl zu Einstellung Gewichte von NURBS mit KNNs
# TODO: Frage: Batchsize variieren?
# TODO: Frage: Unsupervised learning korrekt verwendet?






