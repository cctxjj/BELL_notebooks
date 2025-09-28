# Todo: Idee: Nurb-Surface --> Kurvenpunkte werden jeweils interpoliert, custom metrik --> Nurb macht maybe für Kurve keinen sinn, aber für Fläche
import os
import random
import sys

import numpy as np

from util.datasets.dataset_creator import create_random_bez_points
import util.graphics.visualisations as vis
from aerosandbox import Airfoil, XFoil

from curves.func_based.bézier_curve import bezier_curve
from util.shape_modifier import normalize_points, converge_shape_to_mirrored_airfoil

eval_count = 0
default_path = "C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/neural_curves/airfoil_data"

class DragEvaluator:
    def __init__(self,
                 points: list,
                 range: int = 15,
                 start_angle: int = 0,
                 save_airfoil: bool = True,
                 name_appendix: str = None):
        global eval_count
        eval_count+=1

        # obj attributes
        self.name = f"af_evaluation_{eval_count}_" + name_appendix if name_appendix is not None else f"af_evaluation_{eval_count}"
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
            # Sicherstellen, dass der Zielordner existiert
            save_dir = os.path.join(default_path, self.name)

            vis.visualize_curve(points=points, save_path=save_dir, file_name="curve.png")
            vis.visualize_curve(points=self.airfoil.coordinates, save_path=save_dir, file_name="airfoil.png")

    def execute(self,
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
        alphas = [*range(self.start_angle, self.start_angle+self.range)]
        cds = []
        for alpha in alphas:
            # output progress
            print(f"\revaluating alpha {(alpha+1)}/{(self.start_angle+self.range)} for {self.name}")
            sys.stdout.flush()
            res = xf.alpha(alpha).get("CD")
            if len(res) == 0:
                cds.append(0)
            else:
                cds.append(res[0])
        print(cds)
        print(alphas)
        vis.plot_alpha_cd_correlation(alphas=alphas, cds=cds, save_path=os.path.join(default_path, self.name), file_name="alpha_cd.png")
        return cds
#TODO: Idee: naca airfoils in Verhalten bei Winkeln analysieren --> sind optimiert --> eigene Kurvenpunkte festlegen, zeigen wie KNN langsam lernt, Verhalten zu kopieren und anzupassen

for i in range(10):
    cont_points = create_random_bez_points(int(random.randint(2, 10)), 0, random.randint(15, 45), 0, random.randint(5, 15))
    curve_points = bezier_curve(cont_points, 50)
    ev = DragEvaluator(curve_points, save_airfoil=True, range=45, start_angle=0, name_appendix="find_border")
    ev.execute()