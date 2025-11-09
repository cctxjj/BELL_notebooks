import random

import numpy as np
from aerosandbox import Airfoil
from scipy.sparse.linalg import tfqmr

from curves.func_based.bézier_curve import bezier_curve
from curves.neural.custom_metrics.drag_evaluation import DragEvaluator
from util.datasets.dataset_creator import create_random_curve_points
from util.shape_modifier import converge_tf_shape_to_mirrored_airfoil
import tensorflow as tf


def predict_drag(model, points):
    points_formated = converge_tf_shape_to_mirrored_airfoil(tf.convert_to_tensor(points), resample_req=400)
    # 1) Batch-Dimension vorne hinzufügen -> (1, N, 2)
    points_formated = tf.expand_dims(points_formated, axis=0)

    return model.predict(points_formated)[0][0]

def compare_predictions(n):
    model = tf.keras.models.load_model("C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/models/cd_prediction_model_4.keras")
    for i in range(n):
        cont_points = create_random_curve_points(5, random.randint(0, 3), random.randint(6, 13), 0,
                                                 random.randint(1, 15))
        points = bezier_curve(cont_points, 200)
        cd_nf = DragEvaluator(points, save_airfoil=False, range=30, start_angle=0,
                                  specification=f"nf_prediction_{i}").get_cd(alpha=0)
        cd_tf = predict_drag(model, points)
        if cd_nf is None:
            continue
        print(f"nf: {cd_nf[0][0]} with conf: {cd_nf[1][0]}, tf: {cd_tf} | Diff: {abs(cd_nf[0][0]-cd_tf)/cd_nf[0][0]}")

compare_predictions(1000)