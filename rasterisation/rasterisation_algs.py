# TODO: Add DDA --> compare; add circle (Midpoint, Bresenham) & compare
def dda_line(x_0, y_0, x_1, y_1):
    if x_0 > x_1:
        x_0, x_1 = x_1, x_0
        y_0, y_1 = y_1, y_0

    m = (y_1 - y_0)/(x_1 - x_0)
    print(m)

    result = []
    cur_x = x_0
    cur_y = y_0
    inc_x = 1
    inc_y = m
    if abs(m) > 1:
        m = 1/m
        inc_x, inc_y = m, inc_x

    while abs(cur_x) < x_1 and abs(cur_y) < y_1:
        cur_x += inc_x
        cur_y += inc_y
        result.append((round(cur_x, 0), round(cur_y, 0)))
    # TODO: Fix
    return result


def bresenham(x_0, y_0, x_1, y_1):
    # calculates straight points from Point A(x_0, y_0) to B(x_1, y_1) according to J. E. Bresenham
    # time complexity: O(n)
    # space complexity: O(1)
    # --> not using floating point operations
    if x_0 > x_1:
        x_0, x_1 = x_1, x_0
        y_0, y_1 = y_1, y_0

    d_x = abs(x_1 - x_0) # TODO: Warum muss hier abs hin? --> Antwort: Weil negativwerte die Fehlerrechnung mit d_y verzerren, Fehlerrechnung funktioniert unabhängig von Richtung gleich --> mathematischen Hintergrund betrachten
    d_y = abs(y_1 - y_0) # check for zero division

    if d_x > d_y:
        f_dir = x_0
        s_dir = y_0

        d_n = 2*d_y - d_x

        result = [(f_dir, s_dir)]
        step = 1 if y_0 < y_1 else -1

        while f_dir < x_1:
            f_dir += 1
            if d_n < 0:
                d_n += 2 * d_y
            else:
                s_dir += step
                d_n += 2 * (d_y - d_x)
            result.append((f_dir, s_dir))
        return result
    else:
        f_dir = y_0
        s_dir = x_0

        d_x, d_y = d_y, d_x

        d_n = 2 * d_y - d_x

        result = [(s_dir,
                   f_dir)]
        step = 1 if y_0 < y_1 else -1
        while f_dir != y_1:
            f_dir += step
            if d_n < 0:
                d_n += 2 * d_y
            else:
                s_dir += 1
                d_n += 2 * (d_y - d_x)
                result.append((s_dir, f_dir))
        return result