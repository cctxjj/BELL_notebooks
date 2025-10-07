import numpy as np
from ipycanvas import Canvas
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

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

    # Konvexe Hülle
    if control_points is not None:
        if show_hull and len(control_points) >= 3:
            conv_hull = ConvexHull(control_points)
            hull_points = np.array(control_points)[conv_hull.vertices]
            polygon = Polygon(hull_points, closed=True, fill=True, color=colors["hull"], alpha=0.3, label="convex hull")
            ax.add_patch(polygon)

    # Kontrollpunkte
    if control_points:
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


