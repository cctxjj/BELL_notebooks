import math

import numpy as np

import util.graphics.visualisations as vis
import curves.func_based.bézier_curve as bez_c

def converge_shape_to_airfoil(
        points: list,
        points_num: int = 1000,
        round_digits: int = 5):
    """
    algorithm to convert an open shape like a curve into a closed one by simply connecting the first and last points;
    using DDA-like procedure
    :param points: points of the shape
    :param points_num: number of points connecting the first and last point
    :param round_digits: digits to which the calculated points should be rounded
    :return:
    """
    result = []
    start_x, start_y = points[0]
    end_x, end_y = points[-1]

    m = (end_y - start_y) / (end_x - start_x)
    cur_x = start_x
    cur_y = start_y

    step = (end_x - start_x) / points_num
    m_per_step = m * step

    for i in range(points_num):
        cur_y += m_per_step
        cur_x += step
        result.append((round(cur_x, round_digits), round(cur_y, round_digits)))

    return [*points[::-1], *result]

def converge_shape_to_mirrored_airfoil(
        points: list,
        round_digits: int = 5):
    """
    creates an XFoil-compatible airfoil from a shape by rotating it if necessary and mirroring it along the x-axis
    :param points: points to mirror
    :param round_digits: digits to which the calculated points should be rounded
    :return:
    """
    # Todo: Check whether rotation is really necessary
    # rotation checks
    #m = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
    #if m != 0:
    #    points = rotate_curve(points, -1*math.degrees(math.atan(m)))

    # normalization
    points = normalize_points(points)
    # mirroring & creating airfoil
    bottom = []
    for point in points[1:]:
        bottom.append((round(point[0], round_digits), round(-1 * point[1], round_digits)))
    raw_result = [*points[::-1], *bottom]
    checked_result = []
    for element in raw_result:
        if element not in checked_result:
            checked_result.append(element)
    return checked_result
    #return checked_result, math.degrees(math.atan(m))

def rotate_curve(points, angle: float):
    # formatting points to start at (0, 0)
    vec_x, vec_y = -points[0][0], -points[0][1]
    points_shifted = []
    for point in points:
        points_shifted.append((point[0] + vec_x, point[1] + vec_y))

    # rotation
    new_points = []
    for x, y in points_shifted:
        x_new = math.cos(math.radians(angle)) * x - math.sin(math.radians(angle)) * y
        y_new = math.sin(math.radians(angle)) * x + math.cos(math.radians(angle)) * y
        new_points.append((x_new, y_new))

    # translating back to the original starting point and returning
    return [(x-vec_x, y-vec_y) for x, y in new_points]


def normalize_points(points):
    x_min, x_max = min(points, key=lambda x: x[0])[0], max(points, key=lambda x: x[0])[0]
    y_min = min(points, key=lambda x: x[1])[1]
    d_x = x_max - x_min
    mod_list = []
    for i, point in enumerate(points):
        mod_list.append([(point[0]-x_min)/d_x, (point[1]-y_min)/d_x])
    return mod_list

def __demonstration_correct_airfoil_conversion__():
    cont_points = [(1, 1), (2, 7), (3, 5), (5, 3)]
    bez_curve = bez_c.bezier_curve(cont_points, 100)

    vis.visualize_curve(bez_curve, cont_points, True)
    rotated_bez_curve = converge_shape_to_mirrored_airfoil(bez_curve, 5)[0]
    vis.visualize_curve(rotated_bez_curve, cont_points, True)