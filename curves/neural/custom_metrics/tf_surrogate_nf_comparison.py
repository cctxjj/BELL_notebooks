import random

import numpy as np

from curves.func_based.bézier_curve import bezier_curve
from curves.neural.custom_metrics.drag_evaluation import DragEvaluator
from util.datasets.dataset_creator import create_random_curve_points
from util.shape_modifier import converge_tf_shape_to_mirrored_airfoil
import tensorflow as tf
import util.graphics.visualisations as vis

"""
Vergleich zwischen dem NeuralFoil-Modell und dem Surrogate TF-Modell, Z. 23 anpassen um TF-Modell auszuwählen (NeuralFoil nutzt standardmäßig "large")
"""

def predict_drag(model, points):
    points_formated = converge_tf_shape_to_mirrored_airfoil(tf.convert_to_tensor(points), resample_req=399)
    points_formated = tf.expand_dims(points_formated, axis=0)

    return model.predict(points_formated)[0][0]

def compare_predictions(n):
    model = tf.keras.models.load_model("C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/models/cd_prediction_model_18.keras")
    print(model.summary())
    points_nf = []
    points_tf = []
    while len(points_tf) < n:
        cont_points = create_random_curve_points(5, random.randint(0, 3), random.randint(6, 13), 0,
                                                 random.randint(1, 15))
        points = bezier_curve(cont_points, 200)
        cd_nf = DragEvaluator(points, save_airfoil=False, range=30, start_angle=0).get_cd(alpha=0)
        if cd_nf is None:
            continue
        cd_tf = predict_drag(model, points)
        points_tf.append(cd_tf)
        points_nf.append(cd_nf[0][0])
        print(f"nf: {cd_nf[0][0]} with conf: {cd_nf[1][0]}, tf: {cd_tf} | Diff: {abs(cd_nf[0][0]-cd_tf)/cd_nf[0][0]}")
    dev = [abs(x) for x in vis.plot_cd_comparison(points_nf=points_nf, points_tf=points_tf)]
    print(f"Maximale Abweichung von {np.max(dev)} mit durchschnittlicher Abweichung von {np.mean(dev)} und Median von {np.median(dev)}")


compare_predictions(50)

