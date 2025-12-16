import random

import numpy as np

import curves.func_based.bézier_curve as bez_c
import util.graphics.visualisations as vis
import curves.neural.purposebased.purpose_based_curve as pb_c
import util.datasets.dataset_creator as dc
from curves.neural.custom_metrics.drag_evaluation import DragEvaluator

"""
Erstellt Vergleich der Metrik (cw-Wert) zwischen zweckbasierter Kurve und Basiskurve (Bézierkurve) 
"""

points_num_for_curve = 200
purpose_based_model_name = "m1_20k_4"
len_comparison = 100

# for NeuralFoil to properly evaluate, x_max-x_min must be > y_m

def compare(points_1, points_2):
    # points_2 is expected to be the lower value
    de_1 = DragEvaluator(points_1).get_cd(boundary=0.5)
    de_2 = DragEvaluator(points_2).get_cd(boundary=0.5)
    if de_1 is not None and de_2 is not None:
        print(f"Diff: {(de_1[0][0]-de_2[0][0])/de_1[0][0]}")
    return (de_1[0][0] - de_2[0][0])/de_1[0][0] if de_1 is not None and de_2 is not None else None

difference = []

while len(difference) < len_comparison:
    print(f"{len(difference)}/{len_comparison}")
    cont_points = dc.create_random_curve_points(6, 0, 15, 0,
                                              6)
    points_bez = bez_c.bezier_curve(cont_points, points_num_for_curve)
    points_pb = pb_c.run_curve(cont_points, purpose_based_model_name, points_num_for_curve)
    comparison_res = compare(points_bez, points_pb)
    if comparison_res is not None:
        difference.append(comparison_res)

print(f"len: {len(difference)}")
print(f"min: {min(difference)}")
print(f"max: {max(difference)}")
print(f"mean: {np.mean(difference)}")
print(f"std: {np.std(difference)}")
vis.plot_differences(difference)
