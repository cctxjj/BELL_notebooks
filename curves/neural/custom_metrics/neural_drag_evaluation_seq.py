import math

import tensorflow as tf
import numpy as np



"""
NN Structure: 
    input: parameter t (on interval [0, 1])
    hidden: 32, 32
    output: degree + 1 values corresponding to the bernstein polynomials values at t
--> Sequential is bound to predetermined degree 

using custom training loop to establish basis for unsupervised learning later on
"""
model_num = int(input("Test iteration num the model should be saved as: "))

# setting up variables
epochs = 100
batchsize = 64


# data setup
data = np.load("C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/datasets/bez_curves_cd_vals_small.npz", allow_pickle = True)
points = data["points"].tolist()
values = data["values"].tolist()
print(len(points))
print(len(values))

print(np.array(points).shape)

assert len(values) == len(points)

dataset = tf.data.Dataset.from_tensor_slices((points, values)).batch(batchsize)

total_len = len(points)
ds_train, ds_test = dataset.take(math.floor(total_len/batchsize*90/100)), dataset.skip(math.floor(total_len/batchsize*90/100))

# TODO: WICHTIG --> nicht anhand von Bezierkurve lernen, sondern von Neuralfoil-Input --> Af-Koordinaten
# model setup

model = tf.keras.Sequential(
    layers = [
        tf.keras.layers.InputLayer(shape = (399, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(1, activation = "linear")
    ]
)

# layers

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5)
model.compile(optimizer = optimizer, loss = tf.keras.losses.MeanSquaredError())

print(model.summary())

model.fit(ds_train, epochs = epochs)

test_loss = model.evaluate(ds_test)

model.save(f"C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/models/cd_prediction_model_{model_num}.keras")






