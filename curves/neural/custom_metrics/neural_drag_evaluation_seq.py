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
epochs = 200
batchsize = 16


# data setup
data = np.load("C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/datasets/naca_dataset.npz", allow_pickle = True)
data_bez = np.load("C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/datasets/bez_curves_afs_cd_vals_small.npz", allow_pickle = True)

points_naca = data["points"].tolist()
values_naca = data["values"].tolist()
points_bez = data_bez["points"].tolist()
values_bez = data_bez["values"].tolist()

points = np.concatenate((points_naca, points_bez), axis=0)
values = np.concatenate((values_naca, values_bez), axis=0)

print(np.shape(points))
print(np.shape(values))

X = np.stack(points).astype(np.float32)            # (N, 399, 2)
Y = np.array(values, dtype=np.float32) # (N, 1)

X_train, X_test = X[:(math.floor(len(X)*0.9))], X[(math.floor(len(X)*0.9)):]
Y_train, Y_test = Y[:(math.floor(len(X)*0.9))], Y[(math.floor(len(X)*0.9)):]

print(np.array(points).shape)

assert len(values) == len(points)

print(np.array(points).shape)

assert len(values) == len(points)

ds_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) \
    .shuffle(buffer_size=len(X_train), seed=17, reshuffle_each_iteration=True) \
    .batch(batchsize) \
    .cache() \
    .prefetch(tf.data.AUTOTUNE)

ds_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)) \
    .batch(batchsize) \
    .cache() \
    .prefetch(tf.data.AUTOTUNE)

# model setup

norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(np.stack(X).astype(np.float32))


model = tf.keras.Sequential(
    layers = [
        tf.keras.layers.InputLayer(shape=(399, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.keras.activations.swish),
        tf.keras.layers.Dense(512, activation=tf.keras.activations.swish),
        tf.keras.layers.Dense(512, activation=tf.keras.activations.swish),
        tf.keras.layers.Dense(512, activation=tf.keras.activations.swish),
        tf.keras.layers.Dense(512, activation=tf.keras.activations.swish),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.swish),
    ]
)
# TODO: norm cds, try new Dataset

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5)
model.compile(optimizer = optimizer, loss = tf.keras.losses.Huber())

print(model.summary())

model.fit(ds_train, epochs = epochs)

test_loss = model.evaluate(ds_test)

model.save(f"C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/models/cd_prediction_model_{model_num}.keras")
# aktuelles Modell ist Nr. 4 (Stand 9.11.25)






