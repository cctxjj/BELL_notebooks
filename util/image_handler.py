import numpy
from PIL import Image
import matplotlib.pyplot as plt

def grab_image(path: str):
    img = Image.open(path)
    gray_img = img.convert("L")  # "L" = 8-Bit Graustufen
    return numpy.array(gray_img) # formats Image to [y][x] --> to be handled during processing

def display_img_array(img: numpy.ndarray):
    dpi = 100  # Auflösung der Matplotlib-Figure in dots per inch
    height, width = img.shape[:2]  # Pixelmaße des Bildes

    # Figure so groß wie das Bild in Pixeln (umgerechnet in Inch)
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()