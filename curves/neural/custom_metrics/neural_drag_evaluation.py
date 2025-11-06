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
batchsize = 1


# data setup
data = np.load("C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/datasets/bez_curves_cd_vals_small.npz", allow_pickle = True)
points = data["points"].tolist()
values = data["values"].tolist()

print(np.array(points).shape)

assert len(values) == len(points)



dataset = tf.data.Dataset.from_tensor_slices((points, cds)).batch(batchsize)

total_len = len(points)
ds_train, ds_test = dataset.take(math.floor(total_len/batchsize*90/100)), dataset.skip(math.floor(total_len/batchsize*90/100))


# model setup

model = tf.keras.Sequential(
    layers = [
        tf.keras.layers.InputLayer(shape = (100, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation = "relu"),
        tf.keras.layers.Dense(32, activation = "relu"),
        tf.keras.layers.Dense(1, activation = "sigmoid")
    ]
)

# layers

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)
model.compile(optimizer = optimizer, loss = tf.keras.losses.MeanSquaredError())

print(model.summary())

model.fit(ds_train, epochs = epochs)

test_loss = model.evaluate(ds_test)
print(f"Test Loss: {test_loss:.4f}")

model.save(f"C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/models/cd_prediction_model_{model_num}.keras")






