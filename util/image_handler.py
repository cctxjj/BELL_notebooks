import numpy
from PIL import Image
import matplotlib.pyplot as plt

def format_greyscale_img(path: str):
    img = Image.open(path)
    gray_img = img.convert("L")  # "L" = 8-Bit Graustufen
    return numpy.array(gray_img) # formats Image to [y][x] --> to be handled during processing

def display_img_array(img: numpy.ndarray):
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()