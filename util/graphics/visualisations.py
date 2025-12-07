import numpy as np
from ipycanvas import Canvas
from IPython.display import display
import matplotlib.pyplot as plt
from keras.src.ops import dtype
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import tensorflow as tf

#TODO: Visuals etwas improven

# Credit: ChatGPT 4.0
def zeichne_pixel(punkte, pixelgröße=10, rand=20):
    """
    Zeichnet ein Koordinatensystem mit eingezeichneten Achsen und stellt gegebene Punkte als 10x10 Pixel große schwarze Quadrate dar.

    :param punkte: Liste von (x, y)-Tupeln
    :param pixelgröße: Größe jedes Punkts in Pixeln (Standard: 10)
    :param rand: Rand in Pixeln um das Koordinatensystem (Standard: 20)
    """
    if not punkte:
        print("Die Punkteliste ist leer.")
        return

    # Bestimme die minimalen und maximalen x- und y-Werte
    min_x = min(x for x, _ in punkte)
    max_x = max(x for x, _ in punkte)
    min_y = min(y for _, y in punkte)
    max_y = max(y for _, y in punkte)

    # Berechne die Breite und Höhe des Koordinatensystems
    breite = (max_x - min_x + 1) * pixelgröße + 2 * rand
    höhe = (max_y - min_y + 1) * pixelgröße + 2 * rand

    # Erstelle das Canvas
    canvas = Canvas(width=breite, height=höhe)

    for x, y in punkte:
        # Transformiere die Koordinaten in Canvas-Koordinaten
        canvas_x = rand + (x - min_x) * pixelgröße
        canvas_y = höhe - rand - (y - min_y) * pixelgröße - pixelgröße
        canvas.fill_style = 'black'
        canvas.fill_rect(canvas_x, canvas_y, pixelgröße, pixelgröße)

    display(canvas)

# Credit: ChatGPT 4.0
# TODO: Replace with new method
def plot_points(points):
    """
    Zeichnet eine Liste von (x, y)-Tupeln als schwarze Quadrate in einem dynamisch angepassten Koordinatensystem.

    Parameters:
    points (list of tuples): Liste der zu zeichnenden Punkte, z. B. [(x1, y1), (x2, y2), ...]
    """
    if not points:
        print("Die Punkteliste ist leer.")
        return

    # Extrahiere x- und y-Koordinaten
    x_coords, y_coords = zip(*points)

    # Erstelle die Grafik
    fig, ax = plt.subplots()

    # Zeichne die Punkte als schwarze Quadrate
    ax.scatter(x_coords, y_coords, s=100, c='black', marker='s')  # s=100 entspricht etwa 10x10 Pixeln

    # Achsenbeschriftungen
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Gitterlinien hinzufügen
    ax.grid(True)

    # Achsenlimits dynamisch anpassen mit etwas Puffer
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Achsen gleich skalieren
    ax.set_aspect('equal', adjustable='box')

    # Plot anzeigen
    plt.show()

def plot_circular_points(points):
    """
    Zeichnet eine Liste von (x, y)-Tupeln als schwarze Quadrate in einem dynamisch angepassten Koordinatensystem.

    Parameters:
    points (list of tuples): Liste der zu zeichnenden Punkte, z. B. [(x1, y1), (x2, y2), ...]
    """
    if not points:
        print("Die Punkteliste ist leer.")
        return

    # Extrahiere x- und y-Koordinaten
    x_coords, y_coords = zip(*points)

    # Erstelle die Grafik
    fig, ax = plt.subplots()

    # Zeichne die Punkte als schwarze Quadrate
    ax.scatter(x_coords, y_coords, c='black')  # s=100 entspricht etwa 10x10 Pixeln

    # Achsenbeschriftungen
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Gitterlinien hinzufügen
    ax.grid(True)

    # Achsenlimits dynamisch anpassen mit etwas Puffer
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Achsen gleich skalieren
    ax.set_aspect('equal', adjustable='box')

    # Plot anzeigen
    plt.show()


# TODO: reformat comments
def visualize_curve(
        points: list,
        control_points: list = [],
        show_hull: bool = True,
        save_path: str = None,
        file_name: str = None,
        design = None):
    """
    Visualisiert:
    - Punkte
    - optional: konvexe Hülle
    - optional: Kontrollpunkte + B-Spline-Kurve
    speichert bei Bedarf den Plot

    Args:
        points: Liste von (x,y)
        control_points: optionale Kontrollpunkte (Liste von (x,y))
        curve: optionale Kurvenpunkte (Liste von (x,y))
        show_hull: True -> konvexe Hülle der Punkte zeichnen
        design: dict mit optionalen Farben/Styles
        :param design:
        :param instant_close:
        :param save_path:
        :param points:
        :param control_points:
        :param show_hull:
        :param save_fig:
    """
    if design is None: design = {}
    colors = {
        "points": design.get("points", "black"),
        "hull": design.get("hull", "tab:gray"),
        "control": design.get("control", "red"),
        "curve": design.get("curve", "tab:blue"),
    }
    fig, ax = plt.subplots()
    #fig.figure(figsize=design.get("figsize", (6, 6)))

    # Konvexe Hülle
    if control_points is not None:
        if show_hull and len(control_points) >= 3:
            conv_hull = ConvexHull(control_points)
            hull_points = np.array(control_points)[conv_hull.vertices]
            polygon = Polygon(hull_points, closed=True, fill=True, color=colors["hull"], alpha=0.3, label="convex hull")
            ax.add_patch(polygon)

    # Kontrollpunkte
    if len(control_points) != 0:
        cx, cy = zip(*control_points)
        ax.plot(cx, cy, "o--", c=colors["control"], label="control points")

    # Punkte
    px, py = zip(*points)
    ax.scatter(px, py, c=colors["points"], label="points")

    plt.axis("auto")
    plt.legend()

    if save_path is not None:
        assert file_name is not None
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + "/" + file_name, dpi=300)
        plt.close()
    else:
        plt.show()



def visualize_tf_curve(
        points: list,
        control_points: list = None,
        show_hull: bool = True,
        save_path: str = None,
        file_name: str = None,
        design = None):
    """
    Visualisiert:
    - Punkte
    - optional: konvexe Hülle
    - optional: Kontrollpunkte + B-Spline-Kurve
    speichert bei Bedarf den Plot

    Args:
        points: Liste von (x,y)
        control_points: optionale Kontrollpunkte (Liste von (x,y))
        curve: optionale Kurvenpunkte (Liste von (x,y))
        show_hull: True -> konvexe Hülle der Punkte zeichnen
        design: dict mit optionalen Farben/Styles
        :param design:
        :param instant_close:
        :param save_path:
        :param points:
        :param control_points:
        :param show_hull:
        :param save_fig:
    """
    if design is None: design = {}
    colors = {
        "points": design.get("points", "black"),
        "hull": design.get("hull", "tab:gray"),
        "control": design.get("control", "red"),
        "curve": design.get("curve", "tab:blue"),
    }
    fig, ax = plt.subplots()
    #fig.figure(figsize=design.get("figsize", (6, 6)))

    points = np.array(tf.convert_to_tensor(points, dtype=tf.float32))
    # Konvexe Hülle
    if control_points is not None:
        control_points = np.array(tf.convert_to_tensor(control_points, dtype=tf.float32))
        if show_hull and len(control_points) >= 3:
            conv_hull = ConvexHull(control_points)
            hull_points = np.array(control_points)[conv_hull.vertices]
            polygon = Polygon(hull_points, closed=True, fill=True, color=colors["hull"], alpha=0.3, label="convex hull")
            ax.add_patch(polygon)

    # Kontrollpunkte
    if len(control_points) != 0:
        cx, cy = zip(*control_points)
        ax.plot(cx, cy, "o--", c=colors["control"], label="control points")

    # Punkte
    px, py = zip(*points)
    ax.scatter(px, py, c=colors["points"], label="points")

    plt.axis("auto")
    plt.legend()

    if save_path is not None:
        assert file_name is not None
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + "/" + file_name, dpi=300)
        plt.close()
    else:
        plt.show()

import os


def plot_alpha_cd_correlation(alphas:list,
                              cds: list,
                              save_path: str = None,
                              file_name: str = None):
    """
# TODO: comment (credit: ChatGPT)
    """

    plt.figure(figsize=(6, 4))
    plt.scatter(alphas, cds, marker="o", linestyle="-", color="tab:purple", label="$c_w$")
    plt.xlabel("α [°]")
    plt.ylabel("$c_d$")
    plt.title("alpha - c_d relation")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    if save_path is not None:
        assert file_name is not None
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + "/" + file_name, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_runtime(data, alg_title, color="purple", save_path=None, file_name=None):
    """
    # TODO: comment (credit: ChatGPT)
    """
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, marker="o", linestyle="-", color=color, label="f(n)")
    plt.xlabel("Eingabegröße n")
    plt.ylabel("Zeit in Sekunden")
    plt.title("Zeitkomplexität " + alg_title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        assert file_name is not None
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + "/" + file_name, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_runtime_comparison(alg_titles, data, colors=None, title_seperator=" & ", title = None, title_appendix="", save_path=None, file_name=None):
    """
    # TODO: comment (credit: ChatGPT)
    """
    if colors is None:
        colors = ["black", "red"]
    plt.figure(figsize=(6, 4))
    for i in range(len(alg_titles)):
        values = np.array(data[i])
        x = values[:, 0]
        y = values[:, 1]
        plt.scatter(x, y, marker="o", color=colors[i], label=f"f(n) {alg_titles[i]}")

    plt.xlabel("Eingabegröße n")
    plt.ylabel("Zeit in Sekunden")
    plt.title("Vergleich Zeitkomplexität " + title_seperator.join(alg_titles) + title_appendix if title is None else title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        assert file_name is not None
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + "/" + file_name, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_runtime_difference(data1, data2, label1, label2, color1, color2, save_path=None, file_name=None):
    data1_diff = []
    data2_diff = []
    for i in range(len(data1)):
        if data1[i][1] > data2[i][1]:
            data1_diff.append((data1[i][0], (data1[i][1] - data2[i][1])*100/data2[i][1]))
        elif data2[i][1] > data1[i][1]:
            data2_diff.append((data2[i][0], (data2[i][1] - data1[i][1])*100/data1[i][1]))

    data1_diff = np.array(data1_diff)
    data2_diff = np.array(data2_diff)

    plt.figure(figsize=(7, 5))
    if len(data1_diff) > 0:
        plt.scatter(data1_diff[:, 0], data1_diff[:, 1], marker='o', color=color1, label=f"{label1} langsamer")
        plt.plot(data1_diff[:, 0], data1_diff[:, 1], '-', color=color1, alpha=0.5)
    if len(data2_diff) > 0:
        plt.scatter(data2_diff[:, 0], data2_diff[:, 1], marker='s', color=color2, label=f"{label2} langsamer")
        plt.plot(data2_diff[:, 0], data2_diff[:, 1], '-', color=color2, alpha=0.5)

    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Prozentuale Laufzeitunterschiede")
    plt.xlabel("n (Eingabegröße)")
    plt.ylabel("Unterschied in %")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        assert file_name is not None
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + "/" + file_name)
        plt.close()
    else:
        plt.show()


def visualize_tf_points(points: tf.Tensor,
                        title: str | None = None,
                        ax=None,
                        detach: bool = True,
                        marker: str = "-",
                        show_points: bool = False,
                        figsize=(6, 4),
                        equal_aspect: bool = True):
    """
    Visualisiert 2D-Punkte aus einem TF-Tensor mit matplotlib.

    Parameter:
      points: tf.Tensor der Form (N, 2) oder (1, N, 2)
      title: Plot-Titel (optional)
      ax: vorhandene Matplotlib-Achse nutzen (optional)
      detach: True -> Gradientenpfad abkoppeln (tf.stop_gradient), empfohlen für reines Plotten
      marker: Linienstil (z. B. '-', '--', oder '' für nur Scatter)
      show_points: True -> zusätzlich Scatter-Punkte
      figsize: Größe der Figure, wenn ax nicht übergeben wird
      equal_aspect: True -> Achsenverhältnis 1:1

    Rückgabe:
      ax: verwendete Matplotlib-Achse
    """
    if not isinstance(points, tf.Tensor):
        points = tf.convert_to_tensor(points)

    # Optionale Abkopplung vom Gradientenpfad (empfohlen fürs Plotten)
    pts = tf.stop_gradient(points) if detach else tf.identity(points)

    # Batch-Dimension (1, N, 2) -> (N, 2)
    if pts.shape.rank == 3 and pts.shape[0] == 1:
        pts = tf.squeeze(pts, axis=0)

    if pts.shape.rank != 2 or pts.shape[-1] != 2:
        raise ValueError("Erwartet Punkte der Form (N, 2) oder (1, N, 2).")

    # In NumPy kopieren, damit Matplotlib keine TF-Abhängigkeit hält
    np_pts = pts.numpy().copy()
    x = np_pts[:, 0]
    y = np_pts[:, 1]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Linie und optional Punkte
    if marker:
        ax.plot(x, y, marker, label="Kontur")
    if show_points:
        ax.scatter(x, y, s=10, c="tab:blue", alpha=0.8, label="Stützpunkte")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if show_points or marker:
        ax.legend()

    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax


def plot_cd_comparison(points_nf, points_tf, show=True, save_path=None, filename_prefix="cd_compare"):
    """
    Visualisiert zwei c_d-Arrays (points_nf und points_tf) gegen ihren Index (x=Index, y=c_d),
    plottet beide jeweils separat und am Ende die prozentuale Abweichung je Index
    (Basiswert: points_tf; Abweichung kann negativ sein).

    Parameter:
      points_nf: Sequenz oder np.ndarray mit c_d-Werten (z. B. Referenz/NF)
      points_tf:  Sequenz oder np.ndarray mit c_d-Werten (z. B. Surrogat/TF)
      show:       True -> Plots anzeigen
      save_path:  Pfad, unter dem PNG-Dateien gespeichert werden (optional)
      filename_prefix: Dateinamenspräfix für gespeicherte Plots (optional)

    Rückgabe:
      deviation_pct: np.ndarray der prozentualen Abweichungen je Index
                     (100 * (points_nf - points_tf) / points_tf), ggf. mit NaN bei points_tf==0
    """
    points_nf = np.asarray(points_nf, dtype=float)
    points_tf  = np.asarray(points_tf, dtype=float)

    if points_nf.shape != points_tf.shape:
        raise ValueError(f"Längen müssen übereinstimmen: got {points_nf.shape} vs {points_tf.shape}")

    n = len(points_tf)
    x = np.arange(n)

    # Farben (Lila-Design)
    c_nf = "#6a0dad"          # dunkles Lila
    c_tf = "#b19cd9"          # helleres Lila
    c_dev = "#8a2be2"         # blau-lila
    c_zero = "#aa3377"        # Marker für TF==0

    #plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25})

    # 1) points_nf separat
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, points_nf, color=c_nf, marker="o", linewidth=1.6, label="points_nf")
    ax1.plot(x, points_tf, color=c_tf, marker="s", linewidth=1.6, label="points_tf")
    ax1.set_title("c_d tensorflow & neuralfoil Vergleich")
    ax1.set_xlabel("Curve Nr.")
    ax1.set_ylabel("c_d")
    ax1.legend()
    fig1.tight_layout()
    if save_path:
        fig1.savefig(f"{save_path}/{filename_prefix}_points_nf.png", dpi=200, bbox_inches="tight")

    # 3) Abweichung in %
    #    Definition: 100 * (points_nf - points_tf) / points_tf
    deviation_pct = np.full_like(points_nf, np.nan, dtype=float)
    nonzero_mask = points_tf != 0
    deviation_pct[nonzero_mask] = 100.0 * (points_tf[nonzero_mask] - points_nf[nonzero_mask]) / points_nf[nonzero_mask]

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6)
    ax3.plot(x, deviation_pct, color=c_dev, marker="d", linewidth=1.6, label="Abweichung [%] von tf zur nf Evaluationen")

    # Optional: markiere Stellen, an denen points_tf==0 war
    if np.any(~nonzero_mask):
        ax3.scatter(x[~nonzero_mask], np.zeros(np.sum(~nonzero_mask)), color=c_zero, marker="x", label="TF-Basis = 0")

    ax3.set_title("Prozentuale Abweichung")
    ax3.set_xlabel("Curve Nr.")
    ax3.set_ylabel("Abweichung [%]")
    ax3.legend()
    fig3.tight_layout()
    if save_path:
        fig3.savefig(f"{save_path}/{filename_prefix}_deviation_pct.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig1); plt.close(fig3)

    return deviation_pct
