import curves.recreation.bézier_curve as bez_c
import curves.recreation.basis_spline as b_spl
import util.visualisations as vis

control_points = [(1, 2), (2, 4), (3, 2), (5, 1), (6, -2)]

points_spline = b_spl.b_spline(3, control_points, [0, 0, 0, 1, 2, 3, 3, 3], 1000)
points_nurb = b_spl.nurb(3, control_points, [0, 0, 0, 1.1, 1.2, 3, 3, 3], [1, 3, 2, 5, 1], 1000)
points_bezier = bez_c.bezier_curve(control_points, 1000)

vis.visualize_curve(points_spline, control_points, True)
vis.visualize_curve(points_nurb, control_points, True)
vis.visualize_curve(points_bezier, control_points, True)