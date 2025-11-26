# Todo: Idee: Nurb-Surface --> Kurvenpunkte werden jeweils interpoliert, custom metrik --> Nurb macht maybe für Kurve keinen sinn, aber für Fläche
import math
import os
import sys

import numpy as np

import util.graphics.visualisations as vis
from aerosandbox import Airfoil, XFoil
import neuralfoil as nf

from curves.func_based.bézier_curve import bezier_curve
from util.shape_modifier import normalize_points, converge_shape_to_mirrored_airfoil

eval_count = 0
default_path = "/data/airfoils/nf_evaluation"
#default_path = "/root/bell/data/airfoils/airfoil_on_server"

def reset_eval_count():
    global eval_count
    eval_count = 0

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
        #translation = insert converge_shape_to_mirrored_airfoil in case rotation is needed rotate wieder # returns list of points (and angle of rotation if rotation is applied manually)
        #self.rotation = translation[1] # TODO: use rotation data to adjust angle of incoming airstream
        af_points = np.array(converge_shape_to_mirrored_airfoil(points))


        # setup airfoil obj
        try:
            self.airfoil = Airfoil(coordinates=af_points, name=self.name).repanel(n_points_per_side=200)
        except Exception as e:
            self.airfoil = None
        #self.airfoil = Airfoil(name="naca0012")

        if save_airfoil and self.airfoil is not None:
            save_dir = os.path.join(default_path, self.name)

            vis.visualize_curve(points=points, save_path=save_dir, file_name="curve.png")
            vis.visualize_curve(points=self.airfoil.coordinates, save_path=save_dir, file_name="airfoil.png")

# TODO: rewrite to fit NeuralFoil

    def execute(
            self,
            re: float = 1e6):
        """
        executes the evaluation
        :param re: reynolds number, default 1e6; expresses flow around airfoil as laminar or turbulent --> assumed to be turbulent, further info under https://www.numberanalytics.com/blog/reynolds-number-aerospace-guide (28.09.25)
        :return:
        # TODO: Idee: auf logischer Basis mathematische Formel für Evaluation formulieren --> desto steiler Winkel desto weniger relevant --> *1/a oder so?
        """
        #print("starting evaluation for ", self.name, "")
        if self.airfoil is None:
            return None
        alphas = [*range(self.start_angle, self.start_angle + self.range)]
        cds = nf.get_aero_from_airfoil(
             airfoil=self.airfoil,
            Re=re,
            alpha=alphas,
        ).get("CD")
        print(cds)
        # TODO: Include prediction certainty?
        #print(f"\revaluation for {self.name} done\n", end="")
        ##print(cds)
        ##print(alphas)
        #vis.plot_alpha_cd_correlation(alphas=alphas, cds=cds, save_path=os.path.join(default_path, self.name), file_name="alpha_cd.png")

        # calculating custom drag value
        n = len(cds)
        d_v = 0
        for index, alpha in enumerate(alphas):
            d_v += cds[alpha] / (alpha if alpha != 0 else 1)
        d_v = d_v / n
        return d_v

    def get_cd(
            self,
            re: float = 1e6,
            alpha: float = 0):
        """
        executes the evaluation
        :param re: reynolds number, default 1e6; expresses flow around airfoil as laminar or turbulent --> assumed to be turbulent, further info under https://www.numberanalytics.com/blog/reynolds-number-aerospace-guide (28.09.25)
        :return:
        # TODO: Idee: auf logischer Basis mathematische Formel für Evaluation formulieren --> desto steiler Winkel desto weniger relevant --> *1/a oder so?
        """
        #print("starting evaluation for ", self.name, "")
        if self.airfoil is None:
            return None
        res = nf.get_aero_from_airfoil(
            airfoil=self.airfoil,
            Re=re,
            alpha=alpha,
        )
        # TODO: make sure keys[0] is confidence in prediction
        # TODO: clean up for differentiating between dataset-creation and usage
        if res[[*res.keys()][0]] > 0.8:
            return res["CD"], res[[*res.keys()][0]]
        return None

    # LEGACY METHOD, to be ignored --> datasetcreator won't be needed
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

    def get_nf_eval(
            self,
            re: float = 1e6,
            alpha: float = 0):
        """
        executes the evaluation, returns all vals
        :param re: reynolds number, default 1e6; expresses flow around airfoil as laminar or turbulent --> assumed to be turbulent, further info under https://www.numberanalytics.com/blog/reynolds-number-aerospace-guide (28.09.25)
        :return:
        """
        # print("starting evaluation for ", self.name, "")
        if self.airfoil is None:
            return None
        return nf.get_aero_from_airfoil(
            airfoil=self.airfoil,
            Re=re,
            alpha=alpha,
        )

#cont_points = [(0, 0), (0.5, 1.5), (3, 2), (10, 0.5), (11, 0)]
#curve_points = bezier_curve(cont_points, 250)
#ev = DragEvaluator(curve_points, save_airfoil=False, range=30, start_angle=0, specification="custom_drag_test_naca0012")
##print(ev.execute())