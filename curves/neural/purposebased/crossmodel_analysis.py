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

# TODO: add more diagramms (unnormed)

def create_csv(specification: str, model_ids: dict, max_n: int =None):
    x_vals, loss, loss_bez, loss_drag, drag_improvement, bezier_shift, ratio = [], [], [], [], [], [], []
    for id, x_val in zip(model_ids.keys(), model_ids.values()):
        data = grab_model_result_data(id)
        if max_n is not None and x_val > max_n:
            continue
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

    # plotting abs vals
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    # plot 1: ratio to g
    axes[0][0].plot(df["n"].tolist(), df["ratio"], "o-", label="cw/MSE")
    axes[0][0].set_xlabel("Gleichgewichtsfaktor")
    axes[0][0].set_ylabel("absoluter Wert")
    axes[0][0].set_title("Verhältnis")
    axes[0][0].grid(True, alpha=0.3)
    axes[0][0].legend()
    # plot 2: drag_impr to g
    axes[0][1].plot(df["n"].tolist(), df["drag_improvement"], "o-", label="cwv")
    axes[0][1].set_xlabel("Gleichgewichtsfaktor")
    axes[0][1].set_ylabel("Verbesserung")
    axes[0][1].set_title("Entwicklung von cwv nach Gleichgewichtsfaktor")
    axes[0][1].grid(True, alpha=0.3)
    axes[0][1].legend()
    # plot 3: bez_shift to g
    axes[1][0].plot(df["n"].tolist(), df["bezier_shift"], "o-", label="MSEz")
    axes[1][0].set_xlabel("Gleichgewichtsfaktor")
    axes[1][0].set_ylabel("Zunahme")
    axes[1][0].set_title("Entwicklung von MSEz nach Gleichgewichtsfaktor")
    axes[1][0].grid(True, alpha=0.3)
    axes[1][0].legend()
    # plot 4: normalized drag- and mse-loss to g
    axes[1][1].plot(df["n"].tolist(), np.array([(x - min(df["loss_bez"])) / (max(df["loss_bez"]) - min(df["loss_bez"])) for x in df["loss_bez"]]), "o-", label="MSE (normalisiert)")
    axes[1][1].plot(df["n"].tolist(), np.array(
        [(x - min(df["loss_drag"])) / (max(df["loss_drag"]) - min(df["loss_drag"])) for x in df["loss_drag"]]), "o-",
                 label="cw (normalisiert)")
    axes[1][1].set_xlabel("Gleichgewichtsfaktor")
    axes[1][1].set_title("Entwicklung MSE und cw nach Gleichgewichtsfaktor")
    axes[1][1].grid(True, alpha=0.3)
    axes[1][1].legend()


    # normalized plots
    drag_impr_normalized = np.array([(x - min(df["drag_improvement"])) / (max(df["drag_improvement"]) - min(df["drag_improvement"])) for x in df["drag_improvement"]])
    bez_shift_normalized = np.array([(x - min(df["bezier_shift"])) / (max(df["bezier_shift"]) - min(df["bezier_shift"])) for x in df["bezier_shift"]])
    ratio_normalized = np.array([(x - min(df["ratio"])) / (max(df["ratio"]) - min(df["ratio"])) for x in df["ratio"]])

    plt.figure(figsize=(8, 4))
    plt.plot(df["n"].tolist(), drag_impr_normalized, "o-", label="cwv (normalisiert)")
    plt.plot(df["n"].tolist(), bez_shift_normalized, "o-", label="MSEz (normalisiert)")
    plt.plot(df["n"].tolist(), ratio_normalized, "o-", label="cw/MSE (normalisiert)")
    plt.xlabel("Gleichgewichtsfaktor")
    plt.title("Modelverhalten nach Gleichgewichtsfaktor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()

# models arranged as (model, drag_factor) pairs in a dictionary for interpretation by def create_csv()

def _discover_models(prefix: str = "m1_20k_") -> dict[str, float]:

    data_dir = "C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\model_analysis_1"
    if not os.path.isdir(data_dir):
        return {}

    result = {}
    for file in os.listdir(data_dir):
        if not (file.startswith("equilibrium_data_model_") and file.endswith(".csv")):
            continue
        model_id = file[len("equilibrium_data_model_"):-len(".csv")]
        if not model_id.startswith(prefix):
            continue
        try:
            factor = float(file[(len("equilibrium_data_model_") + len(prefix)):-len(".csv")])
            result[model_id] = factor
        except ValueError:
            continue

    # nach Drag-Faktor sortieren
    return dict(sorted(result.items(), key=lambda kv: kv[1]))

# models: automatisch aus Dateien ermitteln (statt statischer Liste)
models = _discover_models(prefix="m1_20k_")

specification = input("spec: ")
#create_csv(specification, models, max_n=15)
path= "C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/model_analysis_1/"
plot_csv(f"{path}crossmodel_analysis_{specification}.csv")
