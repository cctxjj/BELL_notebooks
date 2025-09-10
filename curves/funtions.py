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