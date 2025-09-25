# Todo: Idee: Nurb-Surface --> Kurvenpunkte werden jeweils interpoliert, custom metrik --> Nurb macht maybe für Kurve keinen sinn, aber für Fläche
import numpy as np

import util.visualisations as vis
from aerosandbox import Airfoil, XFoil

from curves.func_based.bézier_curve import bezier_curve
from util.shape_modifier import converge_shape_to_airfoil, normalize_points

cont_points = [(0, 0), (1, 0.5), (2, 0.25), (7, 0)]
curve_points = bezier_curve(cont_points, 100)

points = converge_shape_to_airfoil(curve_points, 100, 3)

points = normalize_points(points)
print(points)

vis.visualize_curve(points, None, False)

af = Airfoil(coordinates=np.array(points), name="test")
af = af.repanel(n_points_per_side=100)
af2 = Airfoil(name="naca0012")
print(af.coordinates)
vis.visualize_curve(af.coordinates)




xf = XFoil(
    airfoil=af2,
    Re=1e6
)
#xf.timeout = None

for i in range(0, 90):
    print(xf.alpha(i).get("CD"))