def converge_shape(
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