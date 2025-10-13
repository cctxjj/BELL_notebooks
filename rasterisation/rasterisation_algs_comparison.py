import math
import random
import sys
import time
# source decorators: https://www.geeksforgeeks.org/python/decorators-in-python/
from rasterisation_algs import *
import util.graphics.visualisations as vis
# goal: create random incrementing numbers n for rasterisation process of both lines and circles
# try to visualize increase in time effort + differences in algorithms

data_save_path = "C:\\Users\\Sebastian\\PycharmProjects\BELL_notebooks/data/algs_comparison/rasterisation/"
run = input("run number: ")
stepsize = 250
upper_boundary = 50000

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        return end - start
    return wrapper

# ----------------------------
# line drawing algorithms
# ----------------------------

# time measure for dda:
i = 1
datapoints_collected_dda = []
while i < upper_boundary:
    print(f"\rrunning dda line drawing for n={i} (n_max={upper_boundary})", end="")
    sys.stdout.flush()
    t = measure_time(dda_line)(0, 0, i, i)
    datapoints_collected_dda.append((i, t))
    i += stepsize
print("\ncompleted dda")

# time measure for bresenham:
i = 1
datapoints_collected_bresenham = []
while i < upper_boundary:
    print(f"\rrunning bresenham line drawing for n={i} (n_max={upper_boundary})", end="")
    sys.stdout.flush()
    t = measure_time(bresenham)(0, 0, i, i)
    datapoints_collected_bresenham.append((i, t))
    i += stepsize
print("\ncompleted bresenham")

vis.plot_runtime(data=datapoints_collected_dda, alg_title="DDA Linienrasterung", color="#7B1FA2", save_path=data_save_path+"run_" + run + "/", file_name="graph_dda.png")
vis.plot_runtime(data=datapoints_collected_bresenham, alg_title="Bresenham Linienrasterung", color="#BA68C8", save_path=data_save_path+"run_" + run + "/", file_name="graph_bresenham.png")
vis.plot_runtime_comparison(alg_titles=["DDA", "Bresenham"], data=[datapoints_collected_dda, datapoints_collected_bresenham], colors=["#7B1FA2", "#BA68C8"], save_path=data_save_path+"run_" + run + "/", file_name="graph_comparison.png")
print("saved plots for line drawing algorithms")

# ----------------------------
# circle drawing algorithms
# ----------------------------

# time measure for midpoint circle:
i = 1
datapoints_collected_midpoint_circle = []
while i < upper_boundary:
    print(f"\rrunning midpoint circle drawing for n={i} (n_max={upper_boundary})", end="")
    sys.stdout.flush()
    t = measure_time(mid_point_circle)(0, 0, i)
    datapoints_collected_midpoint_circle.append((i, t))
    i += stepsize
print("\ncompleted midpoint circle")

# time measure for bresenham:
i = 1
datapoints_collected_bresenham_circle = []
while i < upper_boundary:
    print(f"\rrunning bresenham circle drawing for n={i} (n_max={upper_boundary})", end="")
    sys.stdout.flush()
    t = measure_time(bresenham_circle)(0, 0, i)
    datapoints_collected_bresenham_circle.append((i, t))
    i += stepsize
print("\ncompleted bresenham circle")

vis.plot_runtime(data=datapoints_collected_midpoint_circle, alg_title="Midpoint Circle Kreisrasterung", color="#5E35B1", save_path=data_save_path+"run_" + run + "/", file_name="graph_midpoint_circle.png")
vis.plot_runtime(data=datapoints_collected_bresenham_circle, alg_title="Bresenham Kreisrasterung", color="#00897B", save_path=data_save_path+"run_" + run + "/", file_name="graph_bresenham_circle.png")
vis.plot_runtime_comparison(alg_titles=["Midpoint Circle", "Bresenham (Circle)"], data=[datapoints_collected_midpoint_circle, datapoints_collected_bresenham_circle], colors=["#5E35B1", "#00897B"], save_path=data_save_path+"run_" + run + "/", file_name="graph_circle_comparison.png")
print("saved plots for circle drawing algorithms")

# better visual of circle drawing algorithms
vis.plot_runtime_difference(data1=datapoints_collected_midpoint_circle, data2=datapoints_collected_bresenham_circle, label1="Midpoint Circle", label2="Bresenham (Circle)", color1="#5E35B1", color2="#00897B", save_path=data_save_path+"run_" + run + "/", file_name="graph_circle_difference.png")