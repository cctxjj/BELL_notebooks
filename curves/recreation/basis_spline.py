def basis_function(i: int,
                   k: int,
                   t: float,
                   knot_vector: list):
    # Todo: comment
    if k <= 1:
        if knot_vector[i] <= t < knot_vector[i + 1] or (t == knot_vector[-1] and t == knot_vector[i + 1]):
            return 1
        else:
            return 0
    else:
        den_1 = knot_vector[i + k - 1] - knot_vector[i]
        den_2 = knot_vector[i + k] - knot_vector[i + 1]
        term_1 = 0
        term_2 = 0
        if den_1 != 0:
            term_1 = ((t - knot_vector[i]) / den_1) * basis_function(i, k - 1, t, knot_vector)
        if den_2 != 0:
            term_2 = ((knot_vector[i + k] - t) / den_2) * basis_function(i + 1, k - 1, t, knot_vector)
        return term_1 + term_2

def b_spline(k: int,
             control_points: list,
             knot_vector: list,
             points_num: int = 1000):
    # Todo: comment
    n = len(control_points)

    if len(knot_vector) != n + k:
        raise ValueError("Invalid knot vector: knot vector must be of length n + k")

    if knot_vector[-1] <= knot_vector[0]:
        raise ValueError("Invalid knot vector: upper bound must be > lower bound")

    border_bottom = knot_vector[k - 1]
    border_top = knot_vector[n]

    interval_size = border_top - border_bottom
    curve_points = []

    for point_number in range(points_num):
        t = border_bottom + point_number * interval_size / (points_num-1)
        cur_x = 0.0
        cur_y = 0.0
        for i in range(0, n):
            b_spline_value = basis_function(i, k, t, knot_vector)
            cur_x += b_spline_value * control_points[i][0]
            cur_y += b_spline_value * control_points[i][1]
        curve_points.append((cur_x, cur_y))
    return curve_points

def nurb(
        k: int,
        control_points: list,
        knot_vector: list,
        weights: list,
        points_num: int = 1000):
    # Todo: comment
    n = len(control_points)

    if len(knot_vector) != n + k:
        raise ValueError("Invalid knot vector: knot vector must be of length n + k")

    if knot_vector[-1] <= knot_vector[0]:
        raise ValueError("Invalid knot vector: upper bound must be > lower bound")

    if len(weights) != n:
        raise ValueError("Invalid weights: list must be of length n")

    for weight in weights:
        if weight <= 0:
            raise ValueError("Invalid weights: all weights must be > 0")

    border_bottom = knot_vector[k - 1]
    border_top = knot_vector[n]

    interval_size = border_top - border_bottom
    curve_points = []

    for point_number in range(points_num):
        t = border_bottom + point_number * interval_size / (points_num-1)
        cur_x = 0.0
        cur_y = 0.0
        den = 0.0
        for i in range(0, n):
            b_spline_value = basis_function(i, k, t, knot_vector)
            cur_x += b_spline_value * control_points[i][0] * weights[i]
            cur_y += b_spline_value * control_points[i][1] * weights[i]
            den += b_spline_value * weights[i]
        if den == 0:
            den = 1
        curve_points.append((cur_x/den, cur_y/den))
    return curve_points