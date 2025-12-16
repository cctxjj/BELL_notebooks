import curves.func_based.bézier_curve as bez_c
import curves.func_based.basis_spline as b_spl
import util.graphics.visualisations as vis

control_points = [(0, 0), (4, 10), (5, 3), (11, 14), (15, 7), (20, 1)]

points_spline = b_spl.b_spline(3, control_points, [0, 0, 0, 1, 2, 3, 4, 4, 4], 1000)
points_nurb = b_spl.nurb(3, control_points, [0, 0, 0, 1.1, 1.2, 1.3, 3, 3, 3], [1, 3, 2, 5, 1, 2], 1000)
points_bezier = bez_c.bezier_curve(control_points, 1000)

vis.visualize_curve(points_spline, control_points, True)
vis.visualize_curve(points_nurb, control_points, True)
vis.visualize_curve(points_bezier, control_points, True)