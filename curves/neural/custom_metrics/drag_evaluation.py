# Todo: Idee: Nurb-Surface --> Kurvenpunkte werden jeweils interpoliert, custom metrik --> Nurb macht maybe für Kurve keinen sinn, aber für Fläche
import os
import random
import sys

import numpy as np

import util.graphics.visualisations as vis
from aerosandbox import Airfoil, XFoil

from curves.func_based.bézier_curve import bezier_curve
from util.shape_modifier import normalize_points, converge_shape_to_mirrored_airfoil

eval_count = 0
default_path = "C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/neural_curves/airfoil_data_nomenclat2"

class DragEvaluator:
    def __init__(self,
                 points: list,
                 range: int = 15,
                 start_angle: int = 0,
                 save_airfoil: bool = True,
                 specification: str = None):
        global eval_count
        eval_count+=1

        # obj attributes
        self.name = f"{specification}_af_evaluation_{eval_count}" if specification is not None else f"af_evaluation_{eval_count}"
        self.range = range
        self.start_angle = start_angle
        self.save_airfoil = save_airfoil

        # format points
        af_points = normalize_points(points)
        af_points = converge_shape_to_mirrored_airfoil(af_points)
        af_points = np.array(af_points)

        # setup airfoil obj
        self.airfoil = Airfoil(coordinates=af_points, name=self.name).repanel(n_points_per_side=200)
        #self.airfoil = Airfoil(name="naca0012")

        if save_airfoil:
            save_dir = os.path.join(default_path, self.name)

            vis.visualize_curve(points=points, save_path=save_dir, file_name="curve.png")
            vis.visualize_curve(points=self.airfoil.coordinates, save_path=save_dir, file_name="airfoil.png")

    def execute(
            self,
            re: float = 1e6):
        """
        executes the evaluation
        :param re: reynolds number, default 1e6; expresses flow around airfoil as laminar or turbulent --> assumed to be turbulent, further info under https://www.numberanalytics.com/blog/reynolds-number-aerospace-guide (28.09.25)
        :return:
        # TODO: Idee: auf logischer Basis mathematische Formel für Evaluation formulieren --> desto steiler Winkel desto weniger relevant --> *1/a oder so?
        """
        xf = XFoil(
            airfoil=self.airfoil,
            Re=re
        )
        xf.timeout = 3
        alphas = [*range(self.start_angle, self.start_angle+self.range)]
        cds = {}
        for alpha in alphas:
            # output progress
            print(f"\revaluating alpha {(alpha+1)}/{(self.start_angle+self.range)} for {self.name}")
            sys.stdout.flush()
            res = xf.alpha(alpha).get("CD")
            if len(res) != 0:
                if res[0] != 0:
                    cds[alpha if alpha!=0 else 1] = res[0]
        #print(cds)
        #print(alphas)
        #vis.plot_alpha_cd_correlation(alphas=alphas, cds=cds, save_path=os.path.join(default_path, self.name), file_name="alpha_cd.png")

        # calculating custom drag value
        n = len(cds)
        d_v = 0
        stability = len(cds)/len(alphas)
        if stability == 0:
            return 0, stability
        for alpha in cds.keys():
            d_v += cds[alpha] / alpha
        d_v = d_v / n
        return d_v, stability
#TODO: Idee: naca airfoils in Verhalten bei Winkeln analysieren --> sind optimiert --> eigene Kurvenpunkte festlegen, zeigen wie KNN langsam lernt, Verhalten zu kopieren und anzupassen

#cont_points = [(0, 0), (0.5, 1.5), (3, 2), (10, 0.5), (11, 0)]
#curve_points = bezier_curve(cont_points, 50)
#ev = DragEvaluator(curve_points, save_airfoil=False, range=30, start_angle=0, name_appendix="custom_drag_test_naca0012")
#print(ev.execute())