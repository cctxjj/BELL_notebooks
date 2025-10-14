import math
import os

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from fill_algs import *
import util.graphics.image_handler as imgutil

def show(a):
    plt.imshow(a, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()

def compare(imgs, titles, cols=2):
    n = len(imgs)
    rows = math.ceil(n / cols)
    figsize = (cols * 3, rows * 3)  # auto-size based on number of images

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
            ax.set_title(titles[i])
            ax.axis("off")
        else:
            ax.remove()

    plt.tight_layout()
    plt.show()

    plt.show()


def save_gif_pillow(frames, path="output.gif", duration=100):
    """
    Speichert eine Sequenz von Graustufenbildern als GIF mit Pillow.

    Args:
        frames: Liste oder Array von 2D-Numpy-Arrays (alle gleich groß)
        path: Pfad zur Ausgabedatei (z. B. "animation.gif")
        duration: Zeit pro Frame in Millisekunden
    """
    pil_frames = [Image.fromarray(f).resize((f.shape[1] * 4, f.shape[0] * 4), resample=Image.NEAREST) for f in frames]

    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        optimize=False,
        loop=0,  # 0 = endlos
        disposal=2
    )

def plot_1():
    # todo: add comments
    img_1 = imgutil.grab_image("imgs//img_1.png")
    four_con_flood_fill_applied = recursive_stackbased_flood_fill_4con(img_1.copy(), (30, 30), 150)
    eight_con_flood_fill_applied = recursive_stackbased_flood_fill_8con(img_1.copy(), (30, 30), 150)
    four_con_boundary_fill_applied = recursive_stackbased_boundary_fill_4con(img_1.copy(), (30, 30), 150, 96)
    eight_con_boundary_fill_applied = recursive_stackbased_boundary_fill_8con(img_1.copy(), (30, 30), 150, 96)

    compare(
        [img_1, four_con_flood_fill_applied, eight_con_flood_fill_applied, four_con_boundary_fill_applied,
         eight_con_boundary_fill_applied],
        ["original", "4-connected flood fill", "8-connected flood fill", "4-connected boundary fill",
         "8-connected boundary fill"]
    )

def gif_recursive_stackbased_flood_fill_8con(
        img: np.ndarray,
        start: tuple,
        new_color: int,
        img_density: int = 100):
    '''
    recursive, blind implementation of the flood fill algorithm using 8-connectedness
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return Array with the pictures  displaying the process of flooding
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    # array with pictures to be transformed into a gif
    result = [np.array(img, copy=True)]
    i = 0

    stack = [start]
    while stack:
        i += 1
        if i % img_density == 0:
            result.append(np.array(img, copy=True))
        cur = stack.pop()
        img[cur[1], cur[0]] = new_color

        if cur[1] + 1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1]+1))
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] == start_col:
                    stack.append((cur[0] + 1, cur[1] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] == start_col:
                    stack.append((cur[0] - 1, cur[1] + 1))

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1] - 1))
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] == start_col:
                    stack.append((cur[0] + 1, cur[1] - 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] == start_col:
                    stack.append((cur[0] - 1, cur[1] - 1))

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                stack.append((cur[0] + 1, cur[1] ))

        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                stack.append((cur[0] - 1, cur[1]))
    result.append(np.array(img, copy=True))
    return result

def gif_recursive_queuebased_flood_fill_8con(
        img: np.ndarray,
        start: tuple,
        new_color: int,
        img_density: int = 100):
    '''
    recursive, blind implementation of the flood fill algorithm using 8-connectedness
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    # setting up gif imgs
    result = [np.array(img, copy=True)]
    i = 0

    queue = [start]
    while queue:
        i+=1
        if i % img_density == 0:
            result.append(np.array(img, copy=True))
            show(img)
        cur = queue.pop(0)
        img[cur[1], cur[0]] = new_color

        if cur[1] + 1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                queue.append((cur[0], cur[1] + 1))
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] == start_col:
                    queue.append((cur[0] + 1, cur[1] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] == start_col:
                    queue.append((cur[0] - 1, cur[1] + 1))

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] == start_col:
                queue.append((cur[0], cur[1] - 1))
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] == start_col:
                    queue.append((cur[0] + 1, cur[1] - 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] == start_col:
                    queue.append((cur[0] - 1, cur[1] - 1))

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                queue.append((cur[0] + 1, cur[1]))

        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                queue.append((cur[0] - 1, cur[1]))

    result.append(np.array(img, copy=True))
    return result


def stack_vs_queuebased_fill_gif(path: str = "demonstration/"):
    os.makedirs(path, exist_ok=True)
    img_1 = imgutil.grab_image("imgs//img_1.png")
    save_gif_pillow(gif_recursive_stackbased_flood_fill_8con(img_1.copy(), (30, 30), 150, 5),
                    path + "stackbased_floodfill_8con.gif", 1)
    save_gif_pillow(gif_recursive_queuebased_flood_fill_8con(img_1.copy(), (30, 30), 150, 50000),
                    path + "queuebased_floodfill_8con.gif", 1)



stack_vs_queuebased_fill_gif()
# vis 1: 4 and 8 con, seed & boundary fill
# vis 2: stack based and queue based fill
# vis 3: recursive & scanline procedure

