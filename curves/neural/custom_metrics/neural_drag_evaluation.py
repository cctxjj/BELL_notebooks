import math
import random

import tensorflow as tf

import sys

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

# setting up variables
epochs = 10
# model setup
model = tf.keras.Sequential(
    layers = [
        tf.keras.layers.InputLayer(shape = (100, 2)),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(1, activation = "sigmoid")
        ]
)
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
model.compile(optimizer = optimizer, loss = "mse")

# data setup
data = np.load("C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/datasets/bez_curves_cd_test_small.npz", allow_pickle = True)
points = data["points"]
values = data["values"]
cds = values[1][::]
alphas = values[0][::]
inputs = list(zip(points, alphas))
print(np.array(cds).shape)
print(np.array(inputs).shape)

total_len = len(inputs)
dataset = tf.data.Dataset.from_tensor_slices((inputs, cds))

ds_train, ds_test = dataset.take(math.floor(total_len*85/100)), dataset.skip(math.floor(total_len*85/100))

model.fit(ds_train, epochs = epochs)
model.predict(ds_test)






