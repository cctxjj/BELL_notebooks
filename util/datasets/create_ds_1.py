import os
import tempfile

from util.datasets.dataset_creator import create_fixed_len_bez_curves_drag_coef_dataset

from aerosandbox import XFoil, Airfoil

print("test0")
tempfile.TemporaryDirectory()
print("test1")
af = Airfoil("naca2412")
print("test2")
xf = XFoil(airfoil=af, verbose=False)
print("test3")
res = xf.alpha(2)
print(res)

length = int(input("Enter desired length of the dataset: "))
n1, n2 = create_fixed_len_bez_curves_drag_coef_dataset(file_name="bez_curves_cd_vals", length=5, max_iterations=1000)
print("done :)")
print(n1, n2)