import tensorflow as tf
import numpy as np

import util.visualisations as vis

import curves.recreation.bézier_curve as bez_c

'''
def loss(y_true, y_pred):
    cor_val = 0
    for i in range(degree + 1):
        bernstein_polynomial_value = bernstein_polynomial(i, degree, t / points_num)
        cor_val += bernstein_polynomial_value * control_points[i][0]
    loss = bernstein_polynomial()
    return loss

'''
# 2 inputs: parameter, control_point i
model = tf.keras.Sequential(
    layers = [
        # Input: num_control_points --> degree (n); 1 int stating which control point is used (i); 1 float stating parameter t
        tf.keras.layers.InputLayer(input_shape = (3,)),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(256, activation = "relu"),
        tf.keras.layers.Dense(528, activation = "relu"),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(1, activation = "relu")
        ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)
model.compile(optimizer = optimizer)


# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

def train_step(neural_net: tf.keras.Sequential, data):
    x = data  # TODO: to be fixed/shortened
    with tf.GradientTape() as tape:
        pred = neural_net(inputs=x, training=True)
        loss = bez_loss(x, pred)
    grads = tape.gradient(loss, neural_net.trainable_variables)
    neural_net.optimizer.apply_gradients(zip(grads, neural_net.trainable_variables))
    return {"loss": loss}


def bez_loss(
        x,
        y_pred
):
    data_batch = x.numpy()
    loss_val = 0
    for data in data_batch:
        cor_val = bez_c.bernstein_polynomial(i=np.int_(data[2]), n=np.int_(data[0]), t=data[1])
        loss_val += abs(y_pred - cor_val) ** 2
    print(loss_val.numpy()[0][0])
    return loss_val

test_data = []
for i in range(32*10):
    n = np.random.random_integers(low=2, high=8)
    test_data.append((n, np.random.random_integers(low=0, high=n), np.random.random()))
print(test_data)

test_data = np.array(test_data, dtype = np.float32)
train_dataset = tf.data.Dataset.from_tensor_slices(test_data)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(1)
for epoche in range(5):
    for step, x in enumerate(train_dataset):
        print(f"step {step}; epoche {epoche}")
        train_step(model, x)

control_points = [(1, 2), (2, 4), (3, 2), (5, 1), (6, -2)]

degree = len(control_points) - 1
curve_points = []
bernstein_vals = []
for i in range(degree + 1):
    current_vals = []
    for t in range(50):
        inp = tf.constant([[degree, i, t / 50]], dtype=tf.float32)
        val = model(inputs=inp).numpy()
        current_vals.append(val[0][0])
    softmaxed_weights = []
    total = sum(current_vals)
    for val in current_vals:
        softmaxed_weights.append(val / total)
        print(val / total)
    bernstein_vals.append(softmaxed_weights)


for t in range(50):
    cur_x = 0
    cur_y = 0
    for i in range(degree + 1):
        bernstein_polynomial_value = bernstein_vals[i][t]
        cur_x += bernstein_polynomial_value * control_points[i][0]
        cur_y += bernstein_polynomial_value * control_points[i][1]
        curve_points.append((cur_x, cur_y))

vis.visualize_curve(curve_points, control_points, True)






