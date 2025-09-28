import sys

import tensorflow as tf
import numpy as np

import util.graphics.visualisations as vis
from curves.funtions import basis_function
from util.data.dataset_creator import create_n_parameter_values


"""
NN Structure: 
    input: parameter t (on interval [0, 1])
    hidden: 32, 32
    output: degree + 1 values corresponding to the bernstein polynomials values at t
--> Sequential is bound to predetermined degree 

using custom training loop to establish basis for unsupervised learning later on
"""

# setting up variables
k = 3
n_control_points = 5
knot_vector = [0, 0, 0, 1, 2, 3, 3, 3]
epochs = 10
samples = 1000

# b-spline-specific vars
border_bottom = knot_vector[k - 1]
border_top = knot_vector[n_control_points]

# Idee: dynamic b-spline --> n Epochen auf b-spline, anschließend auf diverse Parameter parallel in eigenen Epochen
# --> Netz lernt erst Grundform und anschließend Verfeinerung entsprechend definierten Anforderungen
# abwechselnde Belohnung entsprechend Metriken, damit sich verschiedene Parameter nicht in Quere kommen?
# Bézierkurve als einfachere Option für unsupervised learning?

# model setup
model = tf.keras.Sequential(
    layers = [
        tf.keras.layers.InputLayer(shape = (1,)),
        tf.keras.layers.Dense(32, activation = "relu"),
        tf.keras.layers.Dense(32, activation = "relu"),
        tf.keras.layers.Dense(n_control_points, activation = "softmax")
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
        t = float(x.numpy())
        target = np.array([basis_function(i, k, t, knot_vector) for i in range(n_control_points)],
                          dtype=np.float32).reshape(1, -1)
        target = tf.constant(target, dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(tf.subtract(y_pred, target)))

    # Backprop
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# dataset creation and training
train_dataset = create_n_parameter_values(n = 1000, border_bottom = border_bottom, border_top = border_top)

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

for num in range(1000):
    cur_x = 0
    cur_y = 0
    t = border_bottom + num * (border_top-border_bottom) / (1000 - 1)
    pol_vals = model(inputs=tf.constant([[t]], dtype=tf.float32)).numpy()
    for i in range(n_control_points):
        b_spline_value = pol_vals[0][i]
        cur_x += b_spline_value * control_points[i][0]
        cur_y += b_spline_value * control_points[i][1]
    curve_points.append((cur_x, cur_y))

vis.visualize_curve(curve_points, control_points, True)

# TODO: Frage: Wo liegt Effizienz im Ansatz der Wichtung mit KNNs im Vgl zu Einstellung Gewichte von NURBS mit KNNs
# TODO: Frage: Batchsize variieren?
# TODO: Frage: Unsupervised learning korrekt verwendet?






