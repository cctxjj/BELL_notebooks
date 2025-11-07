import math
import random
import sys

import tensorflow as tf
import numpy as np
from aerosandbox import Airfoil

import util.graphics.visualisations as vis
from curves.funtions import bernstein_polynomial
from curves.neural.custom_metrics.drag_evaluation import DragEvaluator
from util.datasets.dataset_creator import create_random_curve_points
from curves.neural.custom_metrics.drag_evaluation import reset_eval_count
from util.shape_modifier import converge_shape_to_mirrored_airfoil

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
epochs = 40
bez_curve_iterations = 4000
cont_points_ds_length = 32

# model setup
model = tf.keras.Sequential(
    layers = [
        tf.keras.layers.InputLayer(shape = (1,)),
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
def train_step(epoche_num: int, drag_pred, data: tf.Tensor=None):
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model(inputs=tf.constant([[t / 200] for t in range(200)], dtype=tf.float32), training=True)
        target_bez_vals = tf.constant(
            np.array([[bernstein_polynomial(i, degree, t / 200) for i in range(degree + 1)] for t in range(200)]), dtype=tf.float32)
        loss_bez = tf.reduce_mean(tf.square(tf.subtract(y_pred, target_bez_vals))) # MSE für Bézierkurve
        if epoche_num >= 1:
            cont_points = data.numpy()

            # calculating curve points
            curve_points = []
            for t in range(200):
                cur_x = 0
                cur_y = 0
                for i in range(degree + 1):
                    func_val = y_pred[t][i]
                    cur_x += func_val * cont_points[i][0]
                    cur_y += func_val * cont_points[i][1]
                curve_points.append((cur_x, cur_y))
            vis.visualize_curve(curve_points, cont_points, True)

            # range loss
            distance_border_points = tf.constant(((curve_points[0][0] - cont_points[0][0]) + (cont_points[-1][0] - curve_points[-1][0]))/2, dtype=tf.float32)
            curve_length = tf.constant(float(curve_points[-1][0] - curve_points[0][0]), dtype=tf.float32)

            point_offsets = []
            for i in range(len(curve_points)-1):
                point_offsets.append(abs(float(curve_points[i+1][0] - curve_points[i][0])-(float(curve_points[-1][0] - curve_points[0][0])/len(curve_points))))
            average_point_offset = tf.maximum(tf.constant(np.sum(np.array(point_offsets))/len(point_offsets), dtype=tf.float32), tf.constant(10e-9))
            # maximum to ensure non-zeroness
            # maybe multiply with 100

            range_loss = tf.divide(tf.multiply(distance_border_points, average_point_offset), curve_length)
            #print(f"range: {range_func_vals}")

            # drag loss calculation
            #drag_evaluation = DragEvaluator(curve_points, specification=f"v7/test1/ep{epoch}").execute()

            # loss calculation
            #loss = tf.multiply(tf.pow(tf.constant(loss_bez, dtype=tf.float32), tf.cast(2*drag_evaluation, dtype=tf.float32)), tf.pow(range_loss, tf.constant(2, dtype=tf.float32)))
            #loss = tf.multiply(
                #tf.cast(drag_evaluation, dtype=tf.float32),
                #tf.pow(range_loss, tf.constant(2, dtype=tf.float32)))
            points_formated = np.array(converge_shape_to_mirrored_airfoil(curve_points))
            points_formated = Airfoil(coordinates=points_formated).repanel(n_points_per_side=200).coordinates

            # 1) Batch-Dimension vorne hinzufügen -> (1, N, 2)
            points_formated = tf.expand_dims(points_formated, axis=0)

            loss = drag_pred.predict(points_formated)[0][0]
            """
            
            curve_points = tf.constant(curve_points, dtype=tf.float32)
            curve_diff = curve_points[1:] - curve_points[:-1]
            curve_length = tf.reduce_sum(tf.sqrt(tf.reduce_sum(curve_diff ** 2, axis=1)))
            loss += 0.001 * (1.0 / curve_length)

            # TODO: zentrales Problem mit Loss: belohnt gegen Punkt konvergierende Kurven
            # --> Stability berücksichtigen, bez_curves irgendwie besser einfließen lassen --> exponentieller Zusammenhang
            """
        else:
            loss = loss_bez
        # Loss calculation
        #loss = drag_curve_squared_loss(model, degree)

    # Backprop
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# dataset creation and training
curve_points = [create_random_curve_points(6, random.randint(0, 3), random.randint(6, 13), 0,
                                                 random.randint(1,
                                                                15)) for i in range(cont_points_ds_length)]
curve_points_ds = tf.data.Dataset.from_tensor_slices(curve_points)


drag_pred = tf.keras.models.load_model("C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/models/cd_prediction_model_1.keras")
for epoch in range(epochs):
    loss = -1
    ind = 0

    # iteration over dataset, training (not using batches)
    if epoch >= 1:
        for data in curve_points_ds:
            loss = train_step(epoch, drag_pred, data)
            ind += 1

            # output progress
            print(f"\rEpoch: {(epoch+1)}/{epochs} | sample {ind}/{cont_points_ds_length} | Loss: {loss}", end="")
            if ind == cont_points_ds_length:
                print(f"\rEpoch: {(epoch+1)}/{epochs} | sample {ind}/{cont_points_ds_length} | Loss: {loss} | epoche completed")
            sys.stdout.flush()
        reset_eval_count()
    else:
        for x in range(bez_curve_iterations):
            loss = train_step(epoch, drag_pred)
            ind += 1

            # output progress
            print(f"\rEpoch: {(epoch + 1)}/{epochs} | Run-through: {ind}/{bez_curve_iterations} | Loss: {loss}", end="")
            if ind == bez_curve_iterations:
                print(f"\rEpoch: {(epoch + 1)}/{epochs} | Run-through: {ind}/{bez_curve_iterations} | Loss: {loss} | epoche completed")
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






