import random
import sys

import tensorflow as tf
import numpy as np

import util.graphics.visualisations as vis
from curves.funtions import bernstein_polynomial
from curves.neural.custom_metrics.drag_evaluation import DragEvaluator
from util.datasets.dataset_creator import create_n_parameter_values, create_random_curve_points


"""
NN Structure: 
    input: parameter t (on interval [0, 1])
    hidden: 32, 32
    output: degree + 1 values corresponding to the bernstein polynomials values at t
--> Sequential is bound to predetermined degree 

using custom training loop to establish basis for unsupervised learning later on
"""

# setting up variables
degree = 5
epochs = 20
epochs_bez_curve = 8

# model setup
model = tf.keras.Sequential(
    layers = [
        tf.keras.layers.InputLayer(shape = (1,)),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(degree + 1, activation = "softmax")
        ]
)
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
model.compile(optimizer = optimizer, loss = "mse")

overall_losses = []

# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
# custom training loop --> unsupervised
def train_step(epoche_num: int):
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model(inputs=tf.constant([[t / 200] for t in range(200)], dtype=tf.float32))
        target_bez_vals = tf.constant(
            np.array([[bernstein_polynomial(i, degree, t / 200) for i in range(degree + 1)] for t in range(200)]), dtype=tf.float32)
        loss = tf.reduce_mean(tf.square(tf.subtract(y_pred, target_bez_vals)))
        cont_points = create_random_curve_points(6, random.randint(0, 3), random.randint(6, 13), 0,
                                                 random.randint(1,
                                                                15))  # TODO: check for effect of switching num_cont_points to random
        if epoche_num >= epochs_bez_curve:
            func_vals = y_pred.numpy()
            curve_points = []
            for t in range(200):
                cur_x = 0
                cur_y = 0
                for i in range(degree + 1):
                    func_val = func_vals[t][i]
                    cur_x += func_val * cont_points[i][0]
                    cur_y += func_val * cont_points[i][1]
                curve_points.append((cur_x, cur_y))
            vis.visualize_curve(curve_points, cont_points, True)
            drag_evaluation = DragEvaluator(curve_points, specification="test_8_training_v3").execute()
            if drag_evaluation[1] == 0:
                return -1
            loss = tf.pow(loss, tf.constant(drag_evaluation[0], dtype=tf.float32))
            overall_losses.append(loss)
            print(f"Overall loss list  : {overall_losses}")

        # Loss calculation
        #loss = drag_curve_squared_loss(model, degree)

    # Backprop
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# dataset creation and training
#train_dataset = create_n_parameter_values(n=samples)

for epoch in range(epochs):
    loss = -1
    ind = 0

    # iteration over dataset, training (not using batches)
    if epoch < epochs_bez_curve:
        for x in range(1000):
            loss = train_step(epoch)
            ind += 1

            # output progress
            print(f"\rEpoche: {(epoch+1)}/{epochs} | Run-through: {ind}/{1000} | Loss: {loss}", end="")
            if ind == 1000:
                print(f"\rEpoch: {(epoch+1)}/{epochs} | Run-through: {ind}/{1000} | Loss: {loss} | epoche completed")
            sys.stdout.flush()
    else:
        for x in range(100):
            loss = train_step(epoch)
            ind += 1

            # output progress
            print(f"\rEpoche: {(epoch + 1)}/{epochs} | Run-through: {ind}/{100} | Loss: {loss}", end="")
            if ind == 100:
                print(f"\rEpoch: {(epoch + 1)}/{epochs} | Run-through: {ind}/{100} | Loss: {loss} | epoche completed")
            sys.stdout.flush()


# Test: creating an example curve
control_points = [(1, 2), (2, 4), (3, 2), (5, 1), (6, -2), (7, 1)]

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
# TODO: DragEvaluator af-Anzeige optimieren






