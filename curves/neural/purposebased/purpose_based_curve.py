import tensorflow as tf
import numpy as np
import util.graphics.visualisations as vis


def run_curve(control_points: list, model_name, num_points: int = 500):
    model = tf.keras.models.load_model("C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\models\\min_drag_curve\\model_" + model_name + ".keras")
    curve_points = []
    func_vals = model.predict(np.array([x / num_points for x in range(num_points)])).tolist()
    for t in range(num_points):
        curve_point_x = 0
        curve_point_y = 0
        for index, control_point in enumerate(control_points):
            curve_point_x += func_vals[t][index] * control_point[0]
            curve_point_y += func_vals[t][index] * control_point[1]
        curve_points.append((curve_point_x, curve_point_y))
    return curve_points

cont_points = [(0, 0), (4, 10), (5, 3), (11, 14), (15, 7), (20, 1)]
points_1 = run_curve(cont_points, "m1_20k_4")
points_2 = run_curve(cont_points, "m1_20k_6")
points_3 = run_curve(cont_points, "m1_20k_10")
points_4 = run_curve(cont_points, "m1_50k_9")
points_5 = run_curve(cont_points, "m1_20k_30")

vis.visualize_curve(points_1, cont_points, True)
vis.visualize_curve(points_2, cont_points, True)
vis.visualize_curve(points_3, cont_points, True)
vis.visualize_curve(points_4, cont_points, True)
vis.visualize_curve(points_5, cont_points, True)

