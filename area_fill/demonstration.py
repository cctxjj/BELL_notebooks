from gifs import *

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

def plot_alg_results():
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

def stack_vs_queuebased_fill_gif(path: str = "demonstration/"):
    os.makedirs(path, exist_ok=True)
    img_1 = imgutil.grab_image("imgs//img_1.png")
    save_gif(gif_recursive_stackbased_flood_fill_8con(img_1.copy(), (30, 30), 150, 5),
             path + "stackbased_floodfill_8con.gif", 1)
    save_gif(gif_recursive_queuebased_flood_fill_8con(img_1.copy(), (30, 30), 150, 50000),
             path + "queuebased_floodfill_8con.gif", 1)


def stackbased_scanline_fill(path: str = "demonstration/"):
    img_1 = imgutil.grab_image("imgs//img_1.png")
    os.makedirs(path, exist_ok = True)
    save_gif(gif_scanline_stackbased_flood_fill_8con(img_1.copy(), (30, 30), 150),
             path + "stackbased_scanline_fill_1.gif", 1)
    img_2 = imgutil.grab_image("imgs//img_2.png")
    os.makedirs(path, exist_ok=True)
    save_gif(gif_scanline_stackbased_flood_fill_8con(img_2.copy(), (30, 30), 150),
             path + "stackbased_scanline_fill_2.gif", 1)

def queuebased_scanline_fill(path: str = "demonstration/"):
    img_1 = imgutil.grab_image("imgs//img_1.png")
    os.makedirs(path, exist_ok = True)
    save_gif(gif_scanline_queuebased_flood_fill_8con(img_1.copy(), (30, 30), 150),
             path + "queuebased_scanline_fill_1.gif", 1)
    img_2 = imgutil.grab_image("imgs//img_2.png")
    os.makedirs(path, exist_ok=True)
    save_gif(gif_scanline_queuebased_flood_fill_8con(img_2.copy(), (30, 30), 150),
             path + "queuebased_scanline_fill_2.gif", 1)

plot_alg_results()
stack_vs_queuebased_fill_gif()
stackbased_scanline_fill()
queuebased_scanline_fill()


