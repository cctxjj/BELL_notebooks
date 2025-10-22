import os
import tempfile

from util.datasets.dataset_creator import create_bez_curves_drag_coef_dataset

length = int(input("Enter desired length of the dataset: "))
n1, n2 = create_bez_curves_drag_coef_dataset(length=length, file_name="bez_curves_cd_vals")
print("done :)")
print(n1, n2)