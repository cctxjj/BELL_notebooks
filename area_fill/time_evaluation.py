import sys
import time

import numpy as np
from PIL import Image

from fill_algs import *
from util.graphics import image_handler as imgutil
from util.graphics import visualisations as vis

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        return end - start
    return wrapper

def show(a):
    plt.imshow(a, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()

def resize(img, factor):
    img_pil = Image.fromarray(img)
    new_size = (round(img_pil.width * factor), round(img_pil.height * factor))
    img_resized = img_pil.resize(new_size)
    return np.array(img_resized)


data_save_path = "C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/algs_comparison/fill/"
run = input("run number: ")
stepsize = 1
upper_boundary = 40

# TODO: Auch Platzkomplexität?
# TODO: Queuebased anpassen

# ----------------------------
# time measurements for 8con on img_1
# ----------------------------

img = img_1 = imgutil.grab_image("imgs//img_1.png")

# time measure for stack-based recursive flood fill:
i = 1
stack_recursive_datapoints = []
while i < upper_boundary:
    print(f"\rrunning stack-based recursive flood fill for n={i}*n0 (n_max={upper_boundary}*n0)", end="")
    sys.stdout.flush()
    img_to_use = resize(img.copy(), i)
    t = measure_time(recursive_stackbased_flood_fill_8con)(img_to_use, (30, 30), 155)
    stack_recursive_datapoints.append((i, t))
    i += stepsize
print("\ncompleted stack-based recursive flood fill")
vis.plot_runtime(data=stack_recursive_datapoints, alg_title="stack-based recursive flood fill", color="#4A0072", save_path=data_save_path+"run_" + run + "/", file_name="graph_stack_based_recursive_flood_fill.png")

# time measure for queue-based recursive flood fill:
i = 1
queue_recursive_datapoints = []
while i < upper_boundary:
    print(f"\rrunning queue-based recursive flood fill for n={i}*n0 (n_max={upper_boundary}*n0)", end="")
    sys.stdout.flush()
    img_to_use = resize(img.copy(), i)
    t = measure_time(recursive_queuebased_flood_fill_8con)(img_to_use, (30, 30), 155)
    queue_recursive_datapoints.append((i, t))
    i += stepsize
print("\ncompleted queue-based recursive flood fill")
vis.plot_runtime(data=queue_recursive_datapoints, alg_title="queue-based recursive flood fill", color="#E91E63", save_path=data_save_path+"run_" + run + "/", file_name="graph_queue_based_recursive_flood_fill.png")

# time measure for stack-based scanline flood fill:
i = 1
stack_scanline_datapoints = []
while i < upper_boundary:
    print(f"\rrunning stack-based scanline flood fill for n={i}*n0 (n_max={upper_boundary}*n0)", end="")
    sys.stdout.flush()
    img_to_use = resize(img.copy(), i)
    t = measure_time(scanline_stackbased_flood_fill_8con)(img_to_use, (30, 30), 155)
    stack_scanline_datapoints.append((i, t))
    i += stepsize
print("\ncompleted stack-based scanline flood fill")
vis.plot_runtime(data=stack_scanline_datapoints, alg_title="stack-based scanline flood fill", color="#9C27B0", save_path=data_save_path+"run_" + run + "/", file_name="graph_stack_based_scanline_flood_fill.png")

# time measure for queue-based scanline flood fill:
i = 1
queue_scanline_datapoints = []
while i < upper_boundary:
    print(f"\rrunning queue-based scanline flood fill for n={i}*n0 (n_max={upper_boundary}*n0)", end="")
    sys.stdout.flush()
    img_to_use = resize(img.copy(), i)
    t = measure_time(scanline_queuebased_flood_fill_8con)(img_to_use, (30, 30), 155)
    queue_scanline_datapoints.append((i, t))
    i += stepsize
print("\ncompleted queue-based scanline flood fill")
vis.plot_runtime(data=queue_scanline_datapoints, alg_title="queue-based scanline flood fill", color="#F06292", save_path=data_save_path+"run_" + run + "/", file_name="graph_queue_based_scanline_flood_fill.png")

# time measure for stack-based scanline flood fill optimized:
stack_scanline_opt_datapoints = []
while i < upper_boundary:
    print(f"\rrunning stack-based scanline flood fill (optimized) for n={i}*n0 (n_max={upper_boundary}*n0)", end="")
    sys.stdout.flush()
    img_to_use = resize(img.copy(), i)
    t = measure_time(scanline_stackbased_flood_fill_8con_optimized)(img_to_use, (30, 30), 155)
    stack_scanline_opt_datapoints.append((i, t))
    i += stepsize
print("\ncompleted stack-based scanline flood fill (optimized)")
vis.plot_runtime(data=stack_scanline_opt_datapoints, alg_title="stack-based scanline flood fill optimized", color="#E1BEE7", save_path=data_save_path+"run_" + run + "/", file_name="graph_stack_based_scanline_flood_fill_opt.png")

# time measure for queue-based scanline flood fill optimized:
queue_scanline_opt_datapoints = []
while i < upper_boundary:
    print(f"\rrunning queue-based scanline flood fill (optimized) for n={i}*n0 (n_max={upper_boundary}*n0)", end="")
    sys.stdout.flush()
    img_to_use = resize(img.copy(), i)
    t = measure_time(scanline_queuebased_flood_fill_8con)(img_to_use, (30, 30), 155)
    queue_scanline_opt_datapoints.append((i, t))
    i += stepsize
print("\ncompleted queue-based scanline flood fill (optimized)")
vis.plot_runtime(data=queue_scanline_opt_datapoints, alg_title="queue-based scanline flood fill optimized", color="#F8BBD0", save_path=data_save_path+"run_" + run + "/", file_name="graph_queue_based_scanline_flood_fill_opt.png")

vis.plot_runtime_comparison(["stack-based recursive flood fill", "queue-based recursive flood fill", "stack-based scanline flood fill", "queue-based scanline flood fill", "stack-based scanline flood fill (optimized)", "queue-based scanline flood fill (optimized)"], [stack_recursive_datapoints, queue_recursive_datapoints, stack_scanline_datapoints, queue_scanline_datapoints, stack_scanline_opt_datapoints, queue_scanline_opt_datapoints], ["#4A0072", "#E91E63", "#9C27B0", "#F06292", "#E1BEE7", "#F8BBD0"], title="Vergleich Zeitkomplexität aller Flood Fill Varianten", save_path=data_save_path+"run_" + run + "/", file_name="graph_comparison_overall.png")
vis.plot_runtime_comparison(["stack-based recursive flood fill", "stack-based scanline flood fill", "stack-based scanline flood fill (optimized)"], [stack_recursive_datapoints, stack_scanline_datapoints, stack_scanline_opt_datapoints], ["#4A0072", "#9C27B0", "#E1BEE7"], title="Vergleich Zeitkomplexität für alle stack-based Varianten", save_path=data_save_path+"run_" + run + "/", file_name="graph_comparison_stack_based.png")
vis.plot_runtime_comparison(["queue-based recursive flood fill", "queue-based scanline flood fill", "queue-based scanline flood fill (optimized)"], [queue_recursive_datapoints, queue_scanline_datapoints, queue_scanline_opt_datapoints], ["#E91E63", "#F06292", "#F8BBD0"], title="Vergleich Zeitkomplexität für alle queue-based Varianten", save_path=data_save_path+"run_" + run + "/", file_name="graph_comparison_queue_based.png")
vis.plot_runtime_difference(data1=stack_scanline_opt_datapoints, data2=queue_scanline_opt_datapoints, label1="stack-based scanline flood fill (optimized)", label2="queue-based scanline flood fill (optimized)", color1="#E1BEE7", color2="#F8BBD0", save_path=data_save_path+"run_" + run + "/", file_name="graph_scanline_opt_difference.png")
vis.plot_runtime_difference(data1=stack_scanline_datapoints, data2=stack_recursive_datapoints, label1="stack-based scanline flood fill", label2="stack-based recursive flood fill", color1="#9C27B0", color2="#4A0072", save_path=data_save_path+"run_" + run + "/", file_name="graph_scanline_recursive_difference.png")
vis.plot_runtime_difference(data1=queue_scanline_datapoints, data2=queue_recursive_datapoints, label1="queue-based scanline flood fill", label2="queue-based recursive flood fill", color1="#F06292", color2="#E91E63", save_path=data_save_path+"run_" + run + "/", file_name="graph_scanline_recursive_difference.png")
vis.plot_runtime_difference(data1=stack_scanline_opt_datapoints, data2=stack_scanline_datapoints, label1="stack-based scanline flood fill (optimized)", label2="stack-based scanline flood fill", color1="#E1BEE7", color2="#9C27B0", save_path=data_save_path+"run_" + run + "/", file_name="graph_scanline_opt_stack_difference.png")
vis.plot_runtime_difference(data1=queue_scanline_opt_datapoints, data2=queue_scanline_datapoints, label1="queue-based scanline flood fill (optimized)", label2="queue-based scanline flood fill", color1="#F8BBD0", color2="#F06292", save_path=data_save_path+"run_" + run + "/", file_name="graph_scanline_opt_queue_difference.png")
vis.plot_runtime_difference(data1=stack_recursive_datapoints, data2=queue_recursive_datapoints, label1="stack-based recursive flood fill", label2="queue-based recursive flood fill", color1="#4A0072", color2="#E91E63", save_path=data_save_path+"run_" + run + "/", file_name="graph_recursive_difference.png")
print("completed time measurements for all algorithms")

