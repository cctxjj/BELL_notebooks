import util.visualisations as vis
from curves.func_based import basis_spline

# TODO: fill in examples
k = 3
n_control_points = 5
knot_vector = [0, 0, 0, 1, 2, 3, 3, 3]

control_points = [(1, 2), (2, 4), (3, 2), (5, 1), (6, -2)]

curve_points = basis_spline.b_spline(k, control_points, knot_vector, 1000)

vis.visualize_curve(curve_points, control_points, True)