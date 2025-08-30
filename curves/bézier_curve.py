import math

def bernstein_polynomial(i: int,
                         n: int,
                         t: float):
    """
    :param i: number of the bernstein polynomial
    :param n: degree of the bernstein polynomial
    :param t: given parameter, curvepoint (on interval [0, 1]) to be calculated at
    :return: Value of the Bernstein Polynomial at point t
    """
    binomial_coefficient = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
    return binomial_coefficient * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(control_points: list,
                 points_num: int = 1000):
    """
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