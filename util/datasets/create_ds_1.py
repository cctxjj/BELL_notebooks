from util.datasets.dataset_creator import create_bez_curves_drag_coef_dataset

"""
Erstellt einen Teil des 1:1 gemischtes Datensatzes zum Training des TF NF-Surrogates (konkret die zufälligen Bez-Kurven)
"""

length = int(input("Enter desired length of the dataset: "))
n1, n2 = create_bez_curves_drag_coef_dataset(length=length, file_name="bez_curves_afs_cd_vals_small")
print("done :)")
print(n1, n2)