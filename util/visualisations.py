from ipycanvas import Canvas
from IPython.display import display
import matplotlib.pyplot as plt

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

