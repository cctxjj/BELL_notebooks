import os

from PIL import Image
from tensorboard.compat.tensorflow_stub.io.gfile import exists

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
    plt.savefig("demonstration/effect_comparison.png")


def save_gif(frames, path="output.gif", duration=100):
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
    save_gif(gif_recursive_stackbased_flood_fill_8con(img_1.copy(), (30, 30), 150, 5),
             path + "stackbased_floodfill_8con.gif", 1)
    save_gif(gif_recursive_queuebased_flood_fill_8con(img_1.copy(), (30, 30), 150, 50000),
             path + "queuebased_floodfill_8con.gif", 1)


def gif_scanline_stackbased_flood_fill_8con(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    '''
    Implementation of the flood fill algorithm using 8-connectedness and the concept of scanlines/runs
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    stack = [start]

    result = [np.array(img, copy=True)]

    def identify_and_fill_run(seed):
        result.append(np.array(img, copy=True))
        img[seed[1], seed[0]] = new_color
        left_b, y = seed[0] - 1, seed[1]
        right_b = left_b + 2

        run_above_detected_on_start = False
        run_below_detected_on_start = False
        if y + 1 < height:
            if img[seed[1] + 1, seed[0]] == start_col:
                stack.append((seed[0], seed[1] + 1))
                run_above_detected_on_start = True
        if y - 1 >= 0:
            if img[seed[1] - 1, seed[0]] == start_col:
                stack.append((seed[0], seed[1] - 1))
                run_below_detected_on_start = True

        # looking for the left boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while left_b >= 0 and img[y, left_b] == start_col:
            img[y, left_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, left_b] != start_col:
                        run_above = False
                elif img[y + 1, left_b] == start_col:
                    run_above = True
                    stack.append((left_b, y + 1))

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, left_b] != start_col:
                        run_below = False
                elif img[y - 1, left_b] == start_col:
                    run_below = True
                    stack.append((left_b, y - 1))
            left_b -= 1

        # checking for 8-connected run on left side
        if left_b >= 0:
            if y - 1 >= 0:
                if img[y - 1, left_b] == start_col:
                    stack.append((left_b, y - 1))
            if y + 1 < height:
                if img[y + 1, left_b] == start_col:
                    stack.append((left_b, y + 1))

        # looking for the right boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while right_b < width and img[y, right_b] == start_col:
            img[y, right_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, right_b] != start_col:
                        run_above = False
                elif img[y + 1, right_b] == start_col:
                    run_above = True
                    stack.append((right_b, y + 1))

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, right_b] != start_col:
                        run_below = False
                elif img[y - 1, right_b] == start_col:
                    run_below = True
                    stack.append((right_b ,y - 1))

            right_b += 1

        # checking for 8-connected run on right side
        if right_b < width:
            if y - 1 >= 0:
                if img[y - 1, right_b] == start_col:
                    stack.append((right_b, y - 1))
            if y + 1 < height:
                if img[y + 1, right_b] == start_col:
                    stack.append((right_b, y + 1))

    while stack:
        cur = stack.pop()
        identify_and_fill_run(cur)
    result.append(np.array(img, copy=True))
    return result

def stackbased_scanline_fill(path: str = "demonstration/"):
    img_1 = imgutil.grab_image("imgs//img_1.png")
    os.makedirs(path, exist_ok = True)
    save_gif(gif_scanline_stackbased_flood_fill_8con(img_1.copy(), (30, 30), 150),
             path + "stackbased_scanline_fill_1.gif", 1)
    img_2 = imgutil.grab_image("imgs//img_2.png")
    os.makedirs(path, exist_ok=True)
    save_gif(gif_scanline_stackbased_flood_fill_8con(img_2.copy(), (30, 30), 150),
             path + "stackbased_scanline_fill_2.gif", 1)

#plot_1()
#stack_vs_queuebased_fill_gif()
stackbased_scanline_fill()
# vis 1: 4 and 8 con, seed & boundary fill
# vis 2: stack based and queue based fill
# vis 3: recursive & scanline procedure

