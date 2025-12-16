import util.graphics.visualisations as vis
from curves.func_based import basis_spline
from curves.func_based import bézier_curve

control_points = [(2, 2), (3, 5), (6, 0), (10, 1), (10, 10)]

k = 3
n_control_points = 5
knot_vector = [0, 0, 0, 1, 2, 3, 3, 3]

curve_points_bez = bézier_curve.bezier_curve(control_points, 1000)
curve_points_bspl = basis_spline.b_spline(k, control_points, knot_vector, 1000)

weights = [1, 4, 2, 2, 1]
curve_points_nurb = basis_spline.nurb(k, control_points, knot_vector, weights,1000)

vis.visualize_curve(curve_points_bez, control_points, True)
vis.visualize_curve(curve_points_bspl, control_points, True)
vis.visualize_curve(curve_points_nurb, control_points, True)