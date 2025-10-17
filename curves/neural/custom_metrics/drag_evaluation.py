# Todo: Idee: Nurb-Surface --> Kurvenpunkte werden jeweils interpoliert, custom metrik --> Nurb macht maybe für Kurve keinen sinn, aber für Fläche
import math
import os
import random
import sys

import numpy as np

import util.graphics.visualisations as vis
from aerosandbox import Airfoil, XFoil

from curves.func_based.bézier_curve import bezier_curve
from util.shape_modifier import normalize_points, converge_shape_to_mirrored_airfoil

eval_count = 0
#default_path = "C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/neural_curves/airfoil_data_nomenclat2"
default_path = "/root/bell/data/neural_curves/airfoil_on_server"

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
        translation = converge_shape_to_mirrored_airfoil(points) # returns list of points and angle of rotation
        self.rotation = translation[1] # TODO: use rotation data to adjust angle of incoming airstream
        af_points = translation[0]
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
            Re=re,
            verbose=False,
            xfoil_repanel=True
            # todo: path anpassen
        )

        xf.timeout = 1000000
        alphas = [*range(self.start_angle+self.rotation, self.start_angle+self.range+self.rotation)]
        cds = {}
        print("\n")
        for ind, alpha in enumerate(alphas):
            # output progress
            print(f"\revaluating alpha {(ind+1)}/{self.range} (={alpha}°/{self.start_angle + self.rotation + self.range}°) for {self.name}", end="")
            sys.stdout.flush()
            res = xf.alpha(alpha).get("CD")
            if len(res) != 0:
                if res[0] != 0:
                    cds[alpha if alpha!=0 else 1] = res[0]
        print(f"\revaluation for {self.name} done\n", end="")
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

    def find_valid_alpha_cd(
            self,
            re: float = 1e8):

        xf = XFoil(
            airfoil=self.airfoil,
            Re=re
        )
        xf.timeout = 3
        alphas = [*range(self.start_angle + math.floor(self.rotation), self.start_angle + self.range + math.floor(self.rotation))]
        for alpha in alphas:
            res = xf.alpha(alpha).get("CD")
            if len(res) != 0:
                if res[0] != 0:
                    return alpha, res[0]
        return None, None



#cont_points = [(0, 0), (0.5, 1.5), (3, 2), (10, 0.5), (11, 0)]
#curve_points = bezier_curve(cont_points, 50)
#ev = DragEvaluator(curve_points, save_airfoil=False, range=30, start_angle=0, name_appendix="custom_drag_test_naca0012")
#print(ev.execute())