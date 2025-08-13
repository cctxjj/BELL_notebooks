# TODO: compare DDA und Bresenham; add circle (Midpoint, Bresenham) & compare
# source used/for further explanation: https://www.tutorialspoint.com/computer_graphics/computer_graphics_dda_algorithm.htm (Last accessed: 12th Aug 2025)
def dda_line(x_0, y_0, x_1, y_1):
    if x_0 > x_1:
        x_0, x_1 = x_1, x_0
        y_0, y_1 = y_1, y_0

    m = (y_1 - y_0) / (x_1 - x_0)
    print(m)

    result = []
    cur_x = x_0
    cur_y = y_0

    if abs(m) > 1:
        if y_1 > y_0:
            m = 1 / m
            while cur_y < y_1:
                result.append((round(cur_x, 0), round(cur_y, 0)))
                cur_x += m
                cur_y += 1
        else:
            m = -1 / m  # TODO: Warum das -1 / m? --> Mathematischer Hintergrund
            while cur_y > y_1:
                result.append((round(cur_x, 0), round(cur_y, 0)))
                cur_x += m
                cur_y -= 1
    else:
        while cur_x != x_1:
            result.append((int(round(cur_x, 0)), int(round(cur_y, 0))))
            cur_x += 1
            cur_y += m
    return result


def bresenham(x_0, y_0, x_1, y_1):
    # calculates straight points from point A(x_0, y_0) to B(x_1, y_1) using procedure by J. E. Bresenham
    # time complexity: O(n)
    # space complexity: O(1)
    # --> not using floating point operations
    if x_0 > x_1:
        x_0, x_1 = x_1, x_0
        y_0, y_1 = y_1, y_0

    d_x = abs(
        x_1 - x_0)  # TODO: Warum muss hier abs hin? --> Antwort: Weil negativwerte die Fehlerrechnung mit d_y verzerren, Fehlerrechnung funktioniert unabhängig von Richtung gleich --> mathematischen Hintergrund betrachten
    d_y = abs(y_1 - y_0)  # TODO: check for zero division

    if d_x > d_y:
        f_dir = x_0
        s_dir = y_0

        d_n = 2 * d_y - d_x

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


def mid_point_circle(x_center, y_center, r):
    x = 0
    y = r
    d = 1.25 - r
    first_oct_x = [x]
    first_oct_y = [y]
    while x <= y:
        x += 1
        if d < 0:
            d += 2 * x + 2
        else:
            y -= 1
            d += 2 * (x - y) + 5
        first_oct_x.append(x)
        first_oct_y.append(y)

    result = [*zip([x + x_center for x in first_oct_x], [y + y_center for y in first_oct_y]),
              *zip([y + y_center for y in first_oct_y], [x + x_center for x in first_oct_x]),
              *zip([y + y_center for y in first_oct_y], [x * -1 + x_center for x in first_oct_x]),
              *zip([x + x_center for x in first_oct_x], [y * -1 + y_center for y in first_oct_y]),
              *zip([x * -1 + x_center for x in first_oct_x], [y * -1 + y_center for y in first_oct_y]),
              *zip([y * -1 + y_center for y in first_oct_y], [x * -1 + x_center for x in first_oct_x]),
              *zip([y * -1 + y_center for y in first_oct_y], [x + x_center for x in first_oct_x]),
              *zip([x * -1 + x_center for x in first_oct_x], [y + y_center for y in first_oct_y])
              ]
    return result

def bresenham_circle(x_center, y_center, r):
    x = 0
    y = r
    d = 3-2*r
    first_oct_x = [x]
    first_oct_y = [y]
    while x <= y:
        x += 1
        if d < 0:
            d += 4 * x + 6
        else:
            y -= 1
            d += 4 * (x - y) + 10
        first_oct_x.append(x)
        first_oct_y.append(y)

    result = [*zip([x + x_center for x in first_oct_x], [y + y_center for y in first_oct_y]),
              *zip([y + y_center for y in first_oct_y], [x + x_center for x in first_oct_x]),
              *zip([y + y_center for y in first_oct_y], [x * -1 + x_center for x in first_oct_x]),
              *zip([x + x_center for x in first_oct_x], [y * -1 + y_center for y in first_oct_y]),
              *zip([x * -1 + x_center for x in first_oct_x], [y * -1 + y_center for y in first_oct_y]),
              *zip([y * -1 + y_center for y in first_oct_y], [x * -1 + x_center for x in first_oct_x]),
              *zip([y * -1 + y_center for y in first_oct_y], [x + x_center for x in first_oct_x]),
              *zip([x * -1 + x_center for x in first_oct_x], [y + y_center for y in first_oct_y])
              ]
    return result


import util.visualisations as vis

vis.plot_points(bresenham_circle(10, 10, 30))
