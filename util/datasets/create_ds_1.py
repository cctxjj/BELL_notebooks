from util.datasets.dataset_creator import create_fixed_len_bez_curves_drag_coef_dataset

from aerosandbox import XFoil, Airfoil
af = Airfoil("naca2412")
xf = XFoil(airfoil=af, verbose=False)
res = xf.alpha(2)
print(res)

length = int(input("Enter desired length of the dataset: "))
n1, n2 = create_fixed_len_bez_curves_drag_coef_dataset(file_name="bez_curves_cd_vals", length=5, max_iterations=1000)
print("done :)")
print(n1, n2)