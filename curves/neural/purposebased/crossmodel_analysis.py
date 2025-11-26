import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def grab_model_result_data(name):
    filename = f"C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\model_analysis_1\\equilibrium_data_model_{name}.csv"
    # CSV laden
    df = pd.read_csv(filename)

    # TODO: ggf. bei fehlender Schlussfolgerungsbasis auch andere enddaten in betracht ziehen --> loss
    loss = df["loss"].tolist()[-1]
    loss_bez = df["loss_bez"].tolist()[-1]
    loss_drag = df["loss_drag"].tolist()[-1]
    ratio = loss_drag / loss_bez

    drag_improvement = df["drag_improvement"].tolist()[-1]
    bezier_shift = df["bezier_shift"].tolist()[-1]

    return {
        "loss": loss,
        "loss_bez": loss_bez,
        "loss_drag": loss_drag,
        "drag_improvement": drag_improvement,
        "bezier_shift": bezier_shift,
        "ratio": ratio
    }

def create_csv(specification: str, model_ids: dict):
    x_vals, loss, loss_bez, loss_drag, drag_improvement, bezier_shift, ratio = [], [], [], [], [], [], []
    for id, x_val in zip(model_ids.keys(), model_ids.values()):
        data = grab_model_result_data(id)
        x_vals.append(x_val)
        loss.append(data["loss"])
        loss_bez.append(data["loss_bez"])
        loss_drag.append(data["loss_drag"])
        drag_improvement.append(data["drag_improvement"])
        bezier_shift.append(data["bezier_shift"])
        ratio.append(data["ratio"])

    path_1 = "C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/model_analysis_1/"
    os.makedirs(path_1, exist_ok=True)
    csv_vals = {
        "n": x_vals,
        "loss": loss,
        "loss_bez": loss_bez,
        "loss_drag": loss_drag,
        "drag_improvement": drag_improvement,
        "bezier_shift": bezier_shift,
        "ratio": ratio
    }
    pd.DataFrame(csv_vals).to_csv(f"{path_1}crossmodel_analysis_{specification}.csv", index=False)

def plot_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 4))
    plt.plot(df["n"].tolist(), df["ratio"], "o-", label="loss_drag / loss_bez")
    plt.xlabel("Drag Faktor")
    plt.ylabel("Verhältnis drag/bez_mse")
    plt.title("Verhältnis von drag zu bezier mse loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    drag_impr_normalized = np.array([(x-min(df["drag_improvement"]))/(max(df["drag_improvement"])-min(df["drag_improvement"])) for x in df["drag_improvement"]])
    bez_shift_normalized = np.array(
        [(x - min(df["bezier_shift"])) / (max(df["bezier_shift"]) - min(df["bezier_shift"])) for x in
         df["bezier_shift"]])
    ratio_normed = np.array(
        [(x - min(df["ratio"])) / (max(df["ratio"]) - min(df["ratio"])) for x in
         df["ratio"]])

    plt.figure(figsize=(8, 4))
    plt.plot(df["n"].tolist(), drag_impr_normalized, "o-", label="drag improvement (normed)")
    plt.plot(df["n"].tolist(), bez_shift_normalized, "o-", label="bez shift (normed)")
    plt.plot(df["n"].tolist(), ratio_normed, "o-", label="drag/bez_mse (normed)")
    plt.xlabel("Drag Faktor")
    plt.title("Modelverhalten nach Drag Faktor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()

# models arranged as (model, drag_factor) pairs in a dictionary for interpretation by def create_csv()
models = {
    "m1_20k_1": 1.0,
    "m1_20k_1.5": 1.5,
    "m1_20k_2": 2.0,
    "m1_20k_2.25": 2.25,
    "m1_20k_2.5": 2.5,
    "m1_20k_2.75": 2.75,
    "m1_20k_3": 3.0,
    "m1_20k_3.25": 3.25,
    "m1_20k_3.5": 3.5,
    "m1_20k_3.75": 3.75,
    "m1_20k_4": 4.0,
    "m1_20k_4.25": 4.25,
    "m1_20k_4.5": 4.5,
    "m1_20k_4.75": 4.75,
    "m1_20k_5": 5.0,
    "m1_20k_5.25": 5.25,
    "m1_20k_5.5": 5.5,
    "m1_20k_5.75": 5.75,
    "m1_20k_6": 6.0,
    "m1_20k_6.25": 6.25,
    "m1_20k_6.5": 6.5,
    "m1_20k_6.75": 6.75,
    "m1_20k_7": 7.0,
    "m1_20k_7.25": 7.25,
    "m1_20k_7.5": 7.5,
    "m1_20k_7.75": 7.75,
    "m1_20k_8_higher_crit": 8.0,
    "m1_20k_8.25": 8.25,
    "m1_20k_8.5": 8.5,
    "m1_20k_8.75": 8.75,
    "m1_20k_9": 9.0,
    "m1_20k_9.25": 9.25,
    "m1_20k_9.5": 9.5,
    "m1_20k_9.75": 9.75,
    "m1_20k_10": 10.0,
    "m1_20k_11": 11.0,
    "m1_20k_12": 12.0,
    "m1_20k_15": 15.0,
    "m1_20k_18": 18.0,
    "m1_20k_21": 21.0,
    "m1_20k_24": 24.0,
    "m1_20k_27": 27.0,
    "m1_20k_30": 30.0,
}
specification = input("spec: ")
create_csv(specification, models)
path= "C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/model_analysis_1/"
plot_csv(f"{path}crossmodel_analysis_{specification}.csv")


