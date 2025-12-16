import numpy
from PIL import Image
import matplotlib.pyplot as plt

def grab_image(path: str):
    img = Image.open(path)
    gray_img = img.convert("L")
    return numpy.array(gray_img) # formats Image to [y][x] --> to be handled during processing

def display_img_array(img: numpy.ndarray):
    dpi = 100
    height, width = img.shape[:2]

    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()