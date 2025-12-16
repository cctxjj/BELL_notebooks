import os

import numpy as np

import util.graphics.visualisations as vis
from aerosandbox import Airfoil, XFoil
import neuralfoil as nf

from util.shape_modifier import converge_shape_to_mirrored_airfoil

"""
Wrapper Klasse, um ausgehend von Kurve (repräsentiert durch Punktmenge points - (x, y)-Paare) durch 
Normalisierung, Spiegelung und Repanel cw-Wert mit NeuralFoil zu errechnen.
Bietet zudem Möglichkeit, reformatiertes Zwischenergebnis (airfoil.coordinates) unter individueller Spezifikation zu speichern (default: False)
"""

eval_count = 0
default_path = "/data/airfoils/nf_evaluation"

def reset_eval_count():
    global eval_count
    eval_count = 0

class DragEvaluator:
    def __init__(self,
                 points: list,
                 range: int = 15,
                 start_angle: int = 0,
                 save_airfoil: bool = False,
                 specification: str = None):
        global eval_count
        eval_count+=1
        # obj attributes
        self.name = f"{specification}_af_evaluation_{eval_count}" if specification is not None else f"af_evaluation_{eval_count}"
        self.range = range
        self.start_angle = start_angle
        self.save_airfoil = save_airfoil

        # format points
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

    def get_cd(
            self,
            re: float = 1e6,
            alpha: float = 0,
            boundary: float = 0.8):
        """
        executes the evaluation
        :param boundary: lower boundary for prediction confidence in order to return a value
        :param alpha: angle of attack for airstream, assumed to be 0
        :param re: reynolds number, default 1e6; expresses flow around airfoil as laminar or turbulent --> assumed to be turbulent, further info under https://www.numberanalytics.com/blog/reynolds-number-aerospace-guide (28.09.25)
        :return:
        """
        #print("starting evaluation for ", self.name, "")
        if self.airfoil is None:
            return None
        res = nf.get_aero_from_airfoil(
            airfoil=self.airfoil,
            Re=re,
            alpha=alpha,
        )
        if res[[*res.keys()][0]] >= boundary:
            return res["CD"], res[[*res.keys()][0]]
        return None

#example usage:
#cont_points = [(0, 0), (0.5, 1.5), (3, 2), (10, 0.5), (11, 0)]
#curve_points = bezier_curve(cont_points, 250)
#ev = DragEvaluator(curve_points, save_airfoil=False, range=30, start_angle=0, specification="custom_drag_test_naca0012")
##print(ev.execute())