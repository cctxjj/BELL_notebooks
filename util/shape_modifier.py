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
    creates an XFoil-compatible airfoil from a shape by mirroring it along the x-axis
    :param points: points to mirror
    :param round_digits: digits to which the calculated points should be rounded
    :return:
    """
    bottom = []
    for point in points[1:]:
        bottom.append((round(point[0], round_digits), round(-1 * point[1], round_digits)))
    return [*points[::-1], *bottom]

def normalize_points(points):
    x_min, x_max = min(points, key=lambda x: x[0])[0], max(points, key=lambda x: x[0])[0]
    y_min = min(points, key=lambda x: x[1])[1]
    d_x = x_max - x_min
    mod_list = []
    for i, point in enumerate(points):
        mod_list.append([(point[0]-x_min)/d_x, (point[1]-y_min)/d_x])
    return mod_list