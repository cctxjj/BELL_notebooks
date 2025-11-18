import math
import random
import sys

import tensorflow as tf
import numpy as np

import util.graphics.visualisations as vis
from curves.funtions import bernstein_polynomial
from util.datasets.dataset_creator import create_random_curve_points
from curves.neural.custom_metrics.drag_evaluation import reset_eval_count
from util.shape_modifier import converge_tf_shape_to_mirrored_airfoil

"""
add nn structure + comments
"""

# setting up variables
degree = 5
bez_curve_iterations = 10000
cont_points_ds_length = 100

# boundaries
upper_boundary = 3000
lower_boundary = 0.6

# model setup
model = tf.keras.Sequential(
    layers = [
        tf.keras.layers.InputLayer(shape = (1,)),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(degree + 1, activation = "softmax")
        ]
)
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
model.compile(optimizer = optimizer)

overall_losses = []

# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
# custom training loop --> unsupervised
def train_step(epoche_num: int, drag_pred, data: tf.Tensor=None, only_calc_loss = False):
    loss = None
    drag_loss = None
    bez_loss = None
    range_loss = None
    with tf.GradientTape() as tape:
        y_pred = model(inputs=tf.constant([[t / 200] for t in range(200)], dtype=tf.float32), training=True)

        target_bez_vals = tf.constant(
            np.array([[bernstein_polynomial(i, degree, t / 200) for i in range(degree + 1)] for t in range(200)]), dtype=tf.float32)
        bez_loss = tf.reduce_mean(tf.square(tf.subtract(y_pred, target_bez_vals))) # MSE für Bézierkurve

        if epoche_num >= 1:
            curve_points = tf.matmul(y_pred, data)
            with tape.stop_recording():
                vis.visualize_tf_curve(curve_points, data, True)
            curve_points = converge_tf_shape_to_mirrored_airfoil(curve_points, resample_req=399)

            #drag loss
            points_formated = tf.expand_dims(curve_points, axis=0)
            drag_loss = drag_pred(points_formated, training=False)[0][0]

            #range loss
            cont_range = tf.subtract(tf.reduce_max(data[:, 0]), tf.reduce_min(data[:, 0]))
            curve_range = tf.subtract(tf.reduce_max(curve_points[:, 0]), tf.reduce_min(curve_points[:, 0]))
            range_loss = tf.divide(cont_range, curve_range)

            #total loss
            loss = tf.add(tf.multiply(bez_loss, tf.constant(1, dtype=tf.float32)),
                          tf.add(tf.multiply(drag_loss, tf.constant(0.0025, dtype=tf.float32)),
                          tf.multiply(range_loss, tf.constant(1, dtype=tf.float32))))

        else:
            loss = bez_loss

    # Backprop
    if only_calc_loss:
        return loss, bez_loss, drag_loss, range_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, bez_loss, drag_loss, range_loss

# dataset creation and training
curve_points = [create_random_curve_points(6, random.randint(0, 3), random.randint(6, 13), 0,
                                                 random.randint(1,
                                                                15)) for i in range(cont_points_ds_length)]
curve_points_ds = tf.data.Dataset.from_tensor_slices(curve_points)


drag_pred = tf.keras.models.load_model("C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/models/cd_prediction_model_17.keras")

crit_met = False
epoch = 0

# crit check vars
drag_0 = None
bez_0 = None

while not crit_met:
    loss = -1
    ind = 0
    average_drag_loss = 0
    if epoch == 1:
        # iteration over dataset, training (not using batches)
        loss_total = []
        loss_bez_total = []
        loss_drag_total = []
        loss_range_total = []
        for data in curve_points_ds:
            loss = train_step(epoch, drag_pred, data, True)
            ind += 1

            # output progress
            loss_total.append(loss[0])
            loss_bez_total.append(loss[1])
            loss_drag_total.append(loss[2])
            loss_range_total.append(loss[3])

            print(
                f"\rEpoch: initial loss evaluation | sample {ind}/{cont_points_ds_length} | Mean loss: {np.mean(loss_total)} with bez: {np.mean(loss_bez_total)},  drag: {np.mean(loss_drag_total)}, range: {np.mean(loss_range_total)}",
                end="")
            if ind == cont_points_ds_length:
                print(f"\rEpoch: initial loss evaluation | sample {ind}/{cont_points_ds_length} | Mean loss: {np.mean(loss_total)} with bez: {np.mean(loss_bez_total)},  drag: {np.mean(loss_drag_total)}, range: {np.mean(loss_range_total)} | epoche completed")
                drag_0 = np.mean(loss_drag_total)
                bez_0 = np.mean(loss_bez_total)
                sys.stdout.flush()
        reset_eval_count()

    elif epoch > 1:
        # iteration over dataset, training (not using batches)
        loss_total = []
        loss_bez_total = []
        loss_drag_total = []
        loss_range_total = []
        for data in curve_points_ds:
            loss = train_step(epoch, drag_pred, data)
            ind += 1

            # output progress
            loss_total.append(loss[0])
            loss_bez_total.append(loss[1])
            loss_drag_total.append(loss[2])
            loss_range_total.append(loss[3])

            print(f"\rEpoch: {epoch} | sample {ind}/{cont_points_ds_length} | Mean loss: {np.mean(loss_total)} with bez: {np.mean(loss_bez_total)},  drag: {np.mean(loss_drag_total)}, range: {np.mean(loss_range_total)}", end="")
            if ind == cont_points_ds_length:
                print(f"\rEpoch: {epoch} | sample {ind}/{cont_points_ds_length} | Mean loss: {np.mean(loss_total)} with bez: {np.mean(loss_bez_total)},  drag: {np.mean(loss_drag_total)}, range: {np.mean(loss_range_total)} | epoche completed")

                # check if criteria are met#

                # lower
                drag_cur = np.mean(loss_drag_total)
                lower_crit = (drag_0 - drag_cur) / drag_0

                # upper
                bez_cur = np.mean(loss_bez_total)
                upper_crit = (bez_cur - bez_0) / bez_0
                # TODO: Wichtig --> Formel --> da nach oben Grenze bei mehreren tausend %

                if lower_crit >= lower_boundary:
                    crit_met = True
                    print(f"Lower criteria met: {lower_crit} >= {lower_boundary} with upper at {upper_crit} | finishing process")
                    continue

                if upper_crit >= upper_boundary:
                    crit_met = True
                    print(f"Upper criteria met: {upper_crit} >= {upper_boundary} with lower at {lower_crit} | finishing process")
                    continue

                print(f"Lower boundary: {lower_crit} | Upper boundary: {upper_crit} | continuing training")

            sys.stdout.flush()
        reset_eval_count()
    else:
        for x in range(bez_curve_iterations):
            loss = train_step(epoch, drag_pred)
            ind += 1
            # output progress
            print(f"\rEpoch: {(epoch + 1)} | Run-through: {ind}/{bez_curve_iterations} | Loss: {loss[0]} (MSE)", end="")
            if ind == bez_curve_iterations:
                print(f"\rEpoch: {(epoch + 1)} | Run-through: {ind}/{bez_curve_iterations} | Loss: {loss[0]} (MSE) | epoche completed")
            sys.stdout.flush()
            # TODO: ggf. oben spacial loss ffür gleichmäßige POunktverteilung hinzufügen
    epoch += 1


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






