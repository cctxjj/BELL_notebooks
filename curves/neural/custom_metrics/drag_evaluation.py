# Todo: Idee: Nurb-Surface --> Kurvenpunkte werden jeweils interpoliert, custom metrik --> Nurb macht maybe für Kurve keinen sinn, aber für Fläche
import numpy as np

import util.visualisations as vis
from aerosandbox import Airfoil, XFoil

from curves.func_based.bézier_curve import bezier_curve
from util.shape_modifier import converge_shape

cont_points = [(0, 0), (1, 2), (2, 3), (7, 0)]
curve_points = bezier_curve(cont_points, 75)

points = converge_shape(curve_points, 75, 3)


vis.visualize_curve(points, cont_points, True)

af = Airfoil(coordinates=np.array(points), name="test")

xf = XFoil(
    airfoil=af,
    Re=1e6
)
#xf.timeout = None

print(xf.alpha(0))
print(xf.alpha(5).get("CD"))