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
epochs = 10
# model setup
curve_input = tf.keras.layers.Input(shape=(100, 2), name='curve_points')
alpha_input = tf.keras.layers.Input(shape=(1,), name='alpha')
curve_flattened = tf.keras.layers.Flatten()(curve_input)

inp = tf.keras.layers.Concatenate()([curve_flattened, alpha_input])

# layers
x = tf.keras.layers.Dense(128, activation="relu")(inp)
x = tf.keras.layers.Dense(128, activation="relu")(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# tf.keras.Model because tf.keras.Sequential can't handle multi input
model = tf.keras.Model(inputs=[curve_input, alpha_input], outputs=output)

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
model.compile(optimizer = optimizer, loss = "mse")

print(model.summary())

# data setup
data = np.load("C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/datasets/bez_curves_cd_test_small.npz", allow_pickle = True)
points = data["points"].tolist()
values = data["values"].tolist()

print(np.array(points).shape)

assert len(values) == len(points)

cds = [x[1] for x in values]
alphas = [x[0] for x in values]

dataset = tf.data.Dataset.from_tensor_slices(({"curve_points": points, "alpha": alphas}, cds)).batch(1)

total_len = len(points)
ds_train, ds_test = dataset.take(math.floor(total_len*85/100)), dataset.skip(math.floor(total_len*85/100))

model.fit(ds_train, epochs = epochs)
model.predict(ds_test)
model.save(f"C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/models/cd_prediction_model_{model_num}.keras")






