from curves.funtions import bernstein_polynomial


def bezier_curve(control_points: list,
                 points_num: int = 1000):
    """
    Note: this is not the optimal construction of a Béziercurve in the sense of efficiency using the de Casteljau
    algorithm but rather a close func_based of the mathematical definition of the curve.
    :param control_points: array of control points in the form of (x, y) tuples
    :param points_num: number of points the intervall [0, 1] should be divided into
    :return: curve points
    """
    degree = len(control_points) - 1
    curve_points = []
    for t in range(points_num):
        cur_x = 0
        cur_y = 0
        for i in range(degree + 1):
            bernstein_polynomial_value = bernstein_polynomial(i, degree, t / points_num)
            cur_x += bernstein_polynomial_value * control_points[i][0]
            cur_y += bernstein_polynomial_value * control_points[i][1]
        curve_points.append((cur_x, cur_y))
    return curve_points