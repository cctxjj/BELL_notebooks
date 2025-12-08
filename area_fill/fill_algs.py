import math
from collections import deque

import numpy as np


def recursive_stackbased_flood_fill_4con(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    """
    recursive, blind implementation of the flood fill algorithm using 4-connectedness
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    """
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")
    stack = [start]
    img[start[1], start[0]] = new_color
    while stack:
        cur = stack.pop()

        if cur[1]+1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1]+1))
                img[cur[1] + 1, cur[0]] = new_color
        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1] - 1))
                img[cur[1] - 1, cur[0]] = new_color
        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                stack.append((cur[0] + 1, cur[1]))
                img[cur[1], cur[0] + 1] = new_color
        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                stack.append((cur[0] - 1, cur[1]))
                img[cur[1], cur[0] - 1] = new_color
    return img

# TODO: irgendwo erwähnen: Designfehler im Algorithmus --> doppelte Prüfung da nicht direkt neue Farbe
# TODO: Farben anpassen

def recursive_stackbased_flood_fill_8con(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    '''
    recursive, blind implementation of the flood fill algorithm using 8-connectedness
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")
    stack = [start]
    img[start[1], start[0]] = new_color
    while stack:
        cur = stack.pop()

        if cur[1] + 1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1]+1))
                img[cur[1] + 1, cur[0]] = new_color
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] == start_col:
                    stack.append((cur[0] + 1, cur[1] + 1))
                    img[cur[1] + 1, cur[0] + 1] = new_color
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] == start_col:
                    stack.append((cur[0] - 1, cur[1] + 1))
                    img[cur[1] + 1, cur[0] - 1] = new_color

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1] - 1))
                img[cur[1] - 1, cur[0]] = new_color
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] == start_col:
                    stack.append((cur[0] + 1, cur[1] - 1))
                    img[cur[1] - 1, cur[0] + 1] = new_color
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] == start_col:
                    stack.append((cur[0] - 1, cur[1] - 1))
                    img[cur[1] - 1, cur[0] - 1] = new_color

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                stack.append((cur[0] + 1, cur[1] ))
                img[cur[1], cur[0] + 1] = new_color

        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                stack.append((cur[0] - 1, cur[1]))
                img[cur[1], cur[0] - 1] = new_color
    return img

def recursive_stackbased_boundary_fill_4con(
        img: np.ndarray,
        start: tuple,
        new_color: int,
        boundary_color: int):
    '''
    recursive, blind implementation of the boundary fill algorithm using 4-connectedness
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :param boundary_color: boundary condition, defines the finite area to be filled
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)

    stack = [start]
    img[start[1], start[0]] = new_color

    while stack:
        cur = stack.pop()

        if cur[1]+1 < height:
            if img[cur[1] + 1, cur[0]] != boundary_color and img[cur[1] + 1, cur[0]] != new_color:
                stack.append((cur[0], cur[1]+1))
                img[cur[1] + 1, cur[0]] = new_color
        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                stack.append((cur[0], cur[1] - 1))
                img[cur[1] - 1, cur[0]] = new_color
        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                stack.append((cur[0] + 1, cur[1]))
                img[cur[1], cur[0] + 1] = new_color
        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                stack.append((cur[0] - 1, cur[1]))
                img[cur[1], cur[0] - 1] = new_color
    return img

def recursive_stackbased_boundary_fill_8con(
        img: np.ndarray,
        start: tuple,
        new_color: int,
        boundary_color: int):
    '''
    recursive, blind implementation of the boundary fill algorithm using 8-connectedness
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :param boundary_color: boundary condition, defines the finite area to be filled
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)

    stack = [start]
    img[start[1], start[0]] = new_color

    while stack:
        cur = stack.pop()

        if cur[1] + 1 < height:
            if img[cur[1] + 1, cur[0]] != boundary_color and img[cur[1] + 1, cur[0]] != new_color:
                stack.append((cur[0], cur[1] + 1))
                img[cur[1] + 1, cur[0]] = new_color
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] != boundary_color and img[cur[1] + 1, cur[0] + 1] != new_color:
                    stack.append((cur[0] + 1, cur[1] + 1))
                    img[cur[1] + 1, cur[0] + 1] = new_color
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] != boundary_color and img[cur[1] + 1, cur[0] - 1] != new_color:
                    stack.append((cur[0] - 1, cur[1] + 1))
                    img[cur[1] + 1, cur[0] - 1] = new_color

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                stack.append((cur[0], cur[1] - 1))
                img[cur[1] - 1, cur[0]] = new_color
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] != boundary_color and img[cur[1] - 1, cur[0] + 1] != new_color:
                    stack.append((cur[0] + 1, cur[1] - 1))
                    img[cur[1] - 1, cur[0] + 1] = new_color
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] != boundary_color and img[cur[1] - 1, cur[0] - 1] != new_color:
                    stack.append((cur[0] - 1, cur[1] - 1))
                    img[cur[1] - 1, cur[0] - 1] = new_color

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                stack.append((cur[0] + 1, cur[1]))
                img[cur[1], cur[0] + 1] = new_color

        if 0 <= cur[0] - 1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                stack.append((cur[0] - 1, cur[1]))
                img[cur[1], cur[0] - 1] = new_color
    return img


def recursive_queuebased_flood_fill_4con(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    '''
    recursive, blind implementation of the flood fill algorithm using 4-connectedness
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    queue = deque([start])
    img[start[1], start[0]] = new_color

    while queue:
        cur = queue.popleft()

        if cur[1]+1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                queue.append((cur[0], cur[1] + 1))
                img[cur[1] + 1, cur[0]] = new_color

        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] == start_col:
                queue.append((cur[0], cur[1] - 1))
                img[cur[1] - 1, cur[0]] = new_color

        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                queue.append((cur[0] + 1, cur[1]))
                img[cur[1], cur[0] + 1] = new_color

        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                queue.append((cur[0] - 1, cur[1]))
                img[cur[1], cur[0] - 1] = new_color
    return img

def recursive_queuebased_flood_fill_8con(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    '''
    recursive, blind implementation of the flood fill algorithm using 8-connectedness
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    queue = deque([start])
    img[start[1], start[0]] = new_color
    while queue:
        cur = queue.popleft()

        if cur[1] + 1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                queue.append((cur[0], cur[1] + 1))
                img[cur[1]+1, cur[0]] = new_color

            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] == start_col:
                    queue.append((cur[0] + 1, cur[1] + 1))
                    img[cur[1]+1, cur[0]+1] = new_color

            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] == start_col:
                    queue.append((cur[0] - 1, cur[1] + 1))
                    img[cur[1]+1, cur[0]-1] = new_color

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] == start_col:
                queue.append((cur[0], cur[1] - 1))
                img[cur[1]-1, cur[0]] = new_color

            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] == start_col:
                    queue.append((cur[0] + 1, cur[1] - 1))
                    img[cur[1]-1, cur[0]+1] = new_color

            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] == start_col:
                    queue.append((cur[0] - 1, cur[1] - 1))
                    img[cur[1]-1, cur[0]-1] = new_color

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                queue.append((cur[0] + 1, cur[1]))
                img[cur[1], cur[0]+1] = new_color

        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                queue.append((cur[0] - 1, cur[1]))
                img[cur[1], cur[0]-1] = new_color
    return img

def recursive_queuebased_boundary_fill_4con(
        img: np.ndarray,
        start: tuple,
        new_color: int,
        boundary_color: int):
    '''
    recursive, blind implementation of the boundary fill algorithm using 4-connectedness
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :param boundary_color: boundary condition, defines the finite area to be filled
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)

    queue = deque([start])
    img[start[1], start[0]] = new_color

    while queue:
        cur = queue.popleft()

        if cur[1]+1 < height:
            if img[cur[1] + 1, cur[0]] != boundary_color and img[cur[1] + 1, cur[0]] != new_color:
                queue.append((cur[0], cur[1] + 1))
                img[cur[1] + 1, cur[0]] = new_color

        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                queue.append((cur[0], cur[1] - 1))
                img[cur[1] - 1, cur[0]] = new_color

        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                queue.append((cur[0] + 1, cur[1]))
                img[cur[1], cur[0] + 1] = new_color

        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                queue.append((cur[0] - 1, cur[1]))
                img[cur[1], cur[0] - 1] = new_color
    return img

def recursive_queuebased_boundary_fill_8con(
        img: np.ndarray,
        start: tuple,
        new_color: int,
        boundary_color: int):
    '''
    recursive, blind implementation of the boundary fill algorithm using 8-connectedness
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :param boundary_color: boundary condition, defines the finite area to be filled
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)

    queue = deque([start])
    img[start[1], start[0]] = new_color

    while queue:
        cur = queue.popleft()

        if cur[1] + 1 < height:
            if img[cur[1] + 1, cur[0]] != boundary_color and img[cur[1] + 1, cur[0]] != new_color:
                queue.append((cur[0], cur[1] + 1))
                img[cur[1] + 1, cur[0]] = new_color

            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] != boundary_color and img[cur[1] + 1, cur[0] + 1] != new_color:
                    queue.append((cur[0] + 1, cur[1] + 1))
                    img[cur[1] + 1, cur[0] + 1] = new_color

            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] != boundary_color and img[cur[1] + 1, cur[0] - 1] != new_color:
                    queue.append((cur[0] - 1, cur[1] + 1))
                    img[cur[1] + 1, cur[0] - 1] = new_color

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                queue.append((cur[0], cur[1] - 1))
                img[cur[1] - 1, cur[0]] = new_color

            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] != boundary_color and img[cur[1] - 1, cur[0] + 1] != new_color:
                    queue.append((cur[0] + 1, cur[1] - 1))
                    img[cur[1] - 1, cur[0] + 1] = new_color

            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] != boundary_color and img[cur[1] - 1, cur[0] - 1] != new_color:
                    queue.append((cur[0] - 1, cur[1] - 1))
                    img[cur[1] - 1, cur[0] - 1] = new_color

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                queue.append((cur[0] + 1, cur[1]))
                img[cur[1], cur[0] + 1] = new_color

        if 0 <= cur[0] - 1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                queue.append((cur[0] - 1, cur[1]))
                img[cur[1], cur[0] - 1] = new_color

    return img

def scanline_stackbased_flood_fill_4con(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    '''
    implementation of the flood fill algorithm using 4-connectedness and the concept of scanlines/runs
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)

    start_col = img[start[1], start[0]]

    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    stack = [start]
    img[start[1], start[0]] = new_color

    def identify_and_fill_run(seed):
        left_b, y = seed[0] - 1, seed[1]
        right_b = left_b + 2

        run_above_detected_on_start = False
        run_below_detected_on_start = False
        if y + 1 < height:
            if img[seed[1] + 1, seed[0]] == start_col:
                stack.append((seed[0], seed[1] + 1))
                img[seed[1] + 1, seed[0]] = new_color
                run_above_detected_on_start = True
        if y - 1 >= 0:
            if img[seed[1] - 1, seed[0]] == start_col:
                stack.append((seed[0], seed[1] - 1))
                img[seed[1] - 1, seed[0]] = new_color
                run_below_detected_on_start = True

        # looking for the left boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while left_b >= 0 and img[y, left_b] == start_col:
            img[y, left_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, left_b] != start_col:
                        run_above = False
                elif img[y + 1, left_b] == start_col:
                    run_above = True
                    stack.append((left_b, y + 1))
                    img[y + 1, left_b] = new_color

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, left_b] != start_col:
                        run_below = False
                elif img[y - 1, left_b] == start_col:
                    run_below = True
                    stack.append((left_b, y - 1))
                    img[y - 1, left_b] = new_color
            left_b -= 1

        # looking for the right boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while right_b < width and img[y, right_b] == start_col:
            img[y, right_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, right_b] != start_col:
                        run_above = False
                elif img[y + 1, right_b] == start_col:
                    run_above = True
                    stack.append((right_b, y + 1))
                    img[y + 1, right_b] = new_color

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, right_b] != start_col:
                        run_below = False
                elif img[y - 1, right_b] == start_col:
                    run_below = True
                    stack.append((right_b ,y - 1))
                    img[y - 1, right_b] = new_color

            right_b += 1

    while stack:
        cur = stack.pop()
        identify_and_fill_run(cur)

    return img

def scanline_stackbased_flood_fill_8con(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    '''
    Implementation of the flood fill algorithm using 8-connectedness and the concept of scanlines/runs
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    stack = [start]
    img[start[1], start[0]] = new_color

    def identify_and_fill_run(seed):
        left_b, y = seed[0] - 1, seed[1]
        right_b = left_b + 2

        run_above_detected_on_start = False
        run_below_detected_on_start = False
        if y + 1 < height:
            if img[seed[1] + 1, seed[0]] == start_col:
                stack.append((seed[0], seed[1] + 1))
                img[seed[1] + 1, seed[0]] = new_color
                run_above_detected_on_start = True
        if y - 1 >= 0:
            if img[seed[1] - 1, seed[0]] == start_col:
                stack.append((seed[0], seed[1] - 1))
                img[seed[1] - 1, seed[0]] = new_color
                run_below_detected_on_start = True

        # looking for the left boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while left_b >= 0 and img[y, left_b] == start_col:
            img[y, left_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, left_b] != start_col:
                        run_above = False
                elif img[y + 1, left_b] == start_col:
                    run_above = True
                    stack.append((left_b, y + 1))
                    img[y + 1, left_b] = new_color

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, left_b] != start_col:
                        run_below = False
                elif img[y - 1, left_b] == start_col:
                    run_below = True
                    stack.append((left_b, y - 1))
                    img[y - 1, left_b] = new_color
            left_b -= 1

        # checking for 8-connected run on left side
        if left_b >= 0:
            if y - 1 >= 0:
                if img[y - 1, left_b] == start_col:
                    stack.append((left_b, y - 1))
                    img[y - 1, left_b] = new_color
            if y + 1 < height:
                if img[y + 1, left_b] == start_col:
                    stack.append((left_b, y + 1))
                    img[y + 1, left_b] = new_color

        # looking for the right boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while right_b < width and img[y, right_b] == start_col:
            img[y, right_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, right_b] != start_col:
                        run_above = False
                elif img[y + 1, right_b] == start_col:
                    run_above = True
                    stack.append((right_b, y + 1))
                    img[y + 1, right_b] = new_color

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, right_b] != start_col:
                        run_below = False
                elif img[y - 1, right_b] == start_col:
                    run_below = True
                    stack.append((right_b ,y - 1))
                    img[y - 1, right_b] = new_color

            right_b += 1

        # checking for 8-connected run on right side
        if right_b < width:
            if y - 1 >= 0:
                if img[y - 1, right_b] == start_col:
                    stack.append((right_b, y - 1))
                    img[y - 1, right_b] = new_color
            if y + 1 < height:
                if img[y + 1, right_b] == start_col:
                    stack.append((right_b, y + 1))
                    img[y + 1, right_b] = new_color

    while stack:
        cur = stack.pop()
        identify_and_fill_run(cur)

    return img

def scanline_queuebased_flood_fill_4con(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    '''
    Implementation of the flood fill algorithm using 4-connectedness and the concept of scanlines/runs
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    queue = deque([start])
    img[start[1], start[0]] = new_color

    def identify_and_fill_run(seed):
        left_b, y = seed[0] - 1, seed[1]
        right_b = left_b + 2

        run_above_detected_on_start = False
        run_below_detected_on_start = False
        if y + 1 < height:
            if img[seed[1] + 1, seed[0]] == start_col:
                queue.append((seed[0], seed[1] + 1))
                img[seed[1] + 1, seed[0]] = new_color
                run_above_detected_on_start = True
        if y - 1 >= 0:
            if img[seed[1] - 1, seed[0]] == start_col:
                queue.append((seed[0], seed[1] - 1))
                img[seed[1] - 1, seed[0]] = new_color
                run_below_detected_on_start = True
        # TODO: Bei 8-connected sicher gehen, dass bei boundary als seed trotzdem ordentlich geprüft wird
        # looking for the left boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while left_b >= 0 and img[y, left_b] == start_col:
            img[y, left_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, left_b] != start_col:
                        run_above = False
                elif img[y + 1, left_b] == start_col:
                    run_above = True
                    queue.append((left_b, y + 1))
                    img[y + 1, left_b] = new_color

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, left_b] != start_col:
                        run_below = False
                elif img[y - 1, left_b] == start_col:
                    run_below = True
                    queue.append((left_b, y - 1))
                    img[y - 1, left_b] = new_color
            left_b -= 1

        # looking for the right boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while right_b < width and img[y, right_b] == start_col:
            img[y, right_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, right_b] != start_col:
                        run_above = False
                elif img[y + 1, right_b] == start_col:
                    run_above = True
                    queue.append((right_b, y + 1))
                    img[y + 1, right_b] = new_color

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, right_b] != start_col:
                        run_below = False
                elif img[y - 1, right_b] == start_col:
                    run_below = True
                    queue.append((right_b ,y - 1))
                    img[y - 1, right_b] = new_color

            right_b += 1

    while queue:
        cur = queue.popleft()
        identify_and_fill_run(cur)

    return img

def scanline_queuebased_flood_fill_8con(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    '''
    Implementation of the flood fill algorithm using 8-connectedness and the concept of scanlines/runs
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    queue = deque([start])
    img[start[1], start[0]] = new_color

    def identify_and_fill_run(seed):
        left_b, y = seed[0] - 1, seed[1]
        right_b = left_b + 2

        run_above_detected_on_start = False
        run_below_detected_on_start = False
        if y + 1 < height:
            if img[seed[1] + 1, seed[0]] == start_col:
                queue.append((seed[0], seed[1] + 1))
                img[seed[1] + 1, seed[0]] = new_color
                run_above_detected_on_start = True
        if y - 1 >= 0:
            if img[seed[1] - 1, seed[0]] == start_col:
                queue.append((seed[0], seed[1] - 1))
                img[seed[1] - 1, seed[0]] = new_color
                run_below_detected_on_start = True

        # looking for the left boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while left_b >= 0 and img[y, left_b] == start_col:
            img[y, left_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, left_b] != start_col:
                        run_above = False
                elif img[y + 1, left_b] == start_col:
                    run_above = True
                    queue.append((left_b, y + 1))
                    img[y + 1, left_b] = new_color

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, left_b] != start_col:
                        run_below = False
                elif img[y - 1, left_b] == start_col:
                    run_below = True
                    queue.append((left_b, y - 1))
                    img[y - 1, left_b] = new_color
            left_b -= 1

        # checking for 8-connected run on left side
        if left_b >= 0:
            if y - 1 >= 0:
                if img[y - 1, left_b] == start_col:
                    queue.append((left_b, y - 1))
                    img[y - 1, left_b] = new_color
            if y + 1 < height:
                if img[y + 1, left_b] == start_col:
                    queue.append((left_b, y + 1))
                    img[y + 1, left_b] = new_color

        # looking for the right boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while right_b < width and img[y, right_b] == start_col:
            img[y, right_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, right_b] != start_col:
                        run_above = False
                elif img[y + 1, right_b] == start_col:
                    run_above = True
                    queue.append((right_b, y + 1))
                    img[y + 1, right_b] = new_color

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, right_b] != start_col:
                        run_below = False
                elif img[y - 1, right_b] == start_col:
                    run_below = True
                    queue.append((right_b ,y - 1))
                    img[y - 1, right_b] = new_color

            right_b += 1

        # checking for 8-connected run on right side
        if right_b < width:
            if y - 1 >= 0:
                if img[y - 1, right_b] == start_col:
                    queue.append((right_b, y - 1))
                    img[y - 1, right_b] = new_color
            if y + 1 < height:
                if img[y + 1, right_b] == start_col:
                    queue.append((right_b, y + 1))
                    img[y + 1, right_b] = new_color

    while queue:
        cur = queue.popleft()
        identify_and_fill_run(cur)

    return img

def scanline_stackbased_flood_fill_4con_optimized(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    '''
    implementation of the flood fill algorithm using 4-connectedness and the concept of scanlines/runs as well as the storage of parental info on the stack
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    stack = [start]
    img[start[1], start[0]] = new_color

    def identify_and_fill_run(seed):

        left_b, y = seed[0] - 1, seed[1]
        right_b = left_b + 2

        parent_above = False
        parent_below = False
        parent_boundary_left = None
        parent_boundary_right = None
        if len(stack) > 0:
            parent_info = stack.pop() # format: (y, left_b, right_b)
            parent_boundary_left = parent_info[1]
            parent_boundary_right = parent_info[2]
            if y + 1 == parent_info[0]:
                parent_above = True
            elif y - 1 == parent_info[0]:
                parent_below = True

        temp_stack = [] # used to store new seed along with parental information later on

        run_above_detected_on_start = False
        run_below_detected_on_start = False
        if not parent_above or not parent_boundary_left <= seed[0] <= parent_boundary_right:
            if y + 1 < height:
                if img[seed[1] + 1, seed[0]] == start_col:
                    temp_stack.append((seed[0], seed[1] + 1))
                    img[seed[1] + 1, seed[0]] = new_color
                    run_above_detected_on_start = True
        if not parent_below or not parent_boundary_left <= seed[0] <= parent_boundary_right:
            if y - 1 >= 0:
                if img[seed[1] - 1, seed[0]] == start_col:
                    temp_stack.append((seed[0], seed[1] - 1))
                    img[seed[1] - 1, seed[0]] = new_color
                    run_below_detected_on_start = True

        # looking for the left boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while left_b >= 0 and img[y, left_b] == start_col:
            img[y, left_b] = new_color

            if not parent_above or not parent_boundary_left <= left_b <= parent_boundary_right:
                if y + 1 < height:
                    if run_above:
                        if img[y + 1, left_b] != start_col:
                            run_above = False
                    elif img[y + 1, left_b] == start_col:
                        run_above = True
                        temp_stack.append((left_b, y + 1))
                        img[y + 1, left_b] = new_color

            if not parent_below or not parent_boundary_left <= left_b <= parent_boundary_right:
                if y - 1 >= 0:
                    if run_below:
                        if img[y - 1, left_b] != start_col:
                            run_below = False
                    elif img[y - 1, left_b] == start_col:
                        run_below = True
                        temp_stack.append((left_b, y - 1))
                        img[y - 1, left_b] = new_color
            left_b -= 1

        # looking for the right boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while right_b < width and img[y, right_b] == start_col:
            img[y, right_b] = new_color

            if not parent_above or not parent_boundary_left <= right_b <= parent_boundary_right:
                if y + 1 < height:
                    if run_above:
                        if img[y + 1, right_b] != start_col:
                            run_above = False
                    elif img[y + 1, right_b] == start_col:
                        run_above = True
                        temp_stack.append((right_b, y + 1))
                        img[y + 1, right_b] = new_color

            if not parent_below or not parent_boundary_left <= right_b <= parent_boundary_right:
                if y - 1 >= 0:
                    if run_below:
                        if img[y - 1, right_b] != start_col:
                            run_below = False
                    elif img[y - 1, right_b] == start_col:
                        run_below = True
                        temp_stack.append((right_b ,y - 1))
                        img[y - 1, right_b] = new_color

            right_b += 1
# TODO: Fix scanline procedures

        run_data = (y, left_b+1, right_b-1)
        for new_seed in temp_stack:
            stack.append(new_seed + run_data)

    while stack:
        cur = stack.pop()
        identify_and_fill_run(cur)

    return img

def scanline_stackbased_flood_fill_8con_optimized(
        img: np.ndarray,
        start: tuple,
        new_color: int):
    '''
    implementation of the flood fill algorithm using 8-connectedness and the concept of scanlines/runs as well as parental info on the stack
    :param img: 2D-Array with ints representing greyscale values from 0 to 255
    :param start: tuple representing starting point P(x|y)
    :param new_color: desired new color to replace old one at P(x|y)
    :return 2D-Array representing the modified picture
    '''
    height, width = np.shape(img)
    start_col = img[start[1], start[0]]
    if start_col == new_color:
        raise ValueError("start_col must be different from new_color")

    stack = [start]
    img[start[1], start[0]] = new_color

    def identify_and_fill_run(seed):

        left_b, y = seed[0] - 1, seed[1]
        right_b = left_b + 2

        parent_above = False
        parent_below = False
        parent_boundary_left = None
        parent_boundary_right = None
        if len(stack) > 0:
            parent_info = stack.pop() # format: (y, left_b, right_b)
            parent_boundary_left = parent_info[1]
            parent_boundary_right = parent_info[2]
            if y + 1 == parent_info[0]:
                parent_above = True
            elif y - 1 == parent_info[0]:
                parent_below = True

        temp_stack = [] # used to store new seed along with parental information later on

        run_above_detected_on_start = False
        run_below_detected_on_start = False
        if not parent_above or not parent_boundary_left <= seed[0] <= parent_boundary_right:
            if y + 1 < height:
                if img[seed[1] + 1, seed[0]] == start_col:
                    temp_stack.append((seed[0], seed[1] + 1))
                    img[seed[1] + 1, seed[0]] = new_color
                    run_above_detected_on_start = True
        if not parent_below or not parent_boundary_left <= seed[0] <= parent_boundary_right:
            if y - 1 >= 0:
                if img[seed[1] - 1, seed[0]] == start_col:
                    temp_stack.append((seed[0], seed[1] - 1))
                    img[seed[1] - 1, seed[0]] = new_color
                    run_below_detected_on_start = True

        # looking for the left boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while left_b >= 0 and img[y, left_b] == start_col:
            img[y, left_b] = new_color

            if not parent_above or not parent_boundary_left <= left_b <= parent_boundary_right:
                if y + 1 < height:
                    if run_above:
                        if img[y + 1, left_b] != start_col:
                            run_above = False
                    elif img[y + 1, left_b] == start_col:
                        run_above = True
                        temp_stack.append((left_b, y + 1))
                        img[y + 1, left_b] = new_color

            if not parent_below or not parent_boundary_left <= left_b <= parent_boundary_right:
                if y - 1 >= 0:
                    if run_below:
                        if img[y - 1, left_b] != start_col:
                            run_below = False
                    elif img[y - 1, left_b] == start_col:
                        run_below = True
                        temp_stack.append((left_b, y - 1))
                        img[y - 1, left_b] = new_color
            left_b -= 1

        # check for 8-connected run on left side
        if left_b >= 0:
            if y - 1 >= 0:
                if img[y - 1, left_b] == start_col:
                    temp_stack.append((left_b, y - 1))
                    img[y - 1, left_b] = new_color
            if y + 1 < height:
                if img[y + 1, left_b] == start_col:
                    temp_stack.append((left_b, y + 1))
                    img[y + 1, left_b] = new_color

        # looking for the right boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while right_b < width and img[y, right_b] == start_col:
            img[y, right_b] = new_color

            if not parent_above or not parent_boundary_left <= right_b <= parent_boundary_right:
                if y + 1 < height:
                    if run_above:
                        if img[y + 1, right_b] != start_col:
                            run_above = False
                    elif img[y + 1, right_b] == start_col:
                        run_above = True
                        temp_stack.append((right_b, y + 1))
                        img[y + 1, right_b] = new_color

            if not parent_below or not parent_boundary_left <= right_b <= parent_boundary_right:
                if y - 1 >= 0:
                    if run_below:
                        if img[y - 1, right_b] != start_col:
                            run_below = False
                    elif img[y - 1, right_b] == start_col:
                        run_below = True
                        temp_stack.append((right_b ,y - 1))
                        img[y - 1, right_b] = new_color

            right_b += 1

        # checking for 8-connected run on right side
        if right_b < width:
            if y - 1 >= 0:
                if img[y - 1, right_b] == start_col:
                    temp_stack.append((right_b, y - 1))
                    img[y - 1, right_b] = new_color
            if y + 1 < height:
                if img[y + 1, right_b] == start_col:
                    temp_stack.append((right_b, y + 1))
                    img[y + 1, right_b] = new_color

        run_data = (y, left_b+1, right_b-1)
        for new_seed in temp_stack:
            stack.append(run_data)
            stack.append(new_seed)

    while stack:
        cur = stack.pop()
        identify_and_fill_run(cur)

    return img

def polygon_fill(
        img: list,
        points: list,
        fill_color: int):
    '''
    Simplified polygon filling algorithm: using dict as edge table, drawing horizontal edges and corners
    :param img: 2D-Array with ints representing greyscale values from 0 to 255, inp image
    :param points: Array containing the tuples of the polygon corners
    :param fill_color: int representing the gray scale value the polygon should be filled with
    :return: 2D-Array with ints representing greyscale values from 0 to 255, modified image
    '''

    height, width = np.shape(img)
    hor_edges = []
    et = {}
    # setting up edge table
    for i, point in enumerate(points):

        x_1, x_2 = point[0], points[(i + 1) % len(points)][0]
        y_1, y_2 = point[1], points[(i + 1) % len(points)][1]

        if x_1 >= width or x_2 >= width or y_1 >= height or y_2 >= height:
            raise ValueError("point must be inside image")

        if y_1 == y_2:
            hor_edges.append((y_1, x_1, x_2))
            continue

        if y_1 > y_2:
            y_min, y_max = y_2, y_1
            x_min, x_max = x_2, x_1
        else:
            y_min, y_max = y_1, y_2
            x_min, x_max = x_1, x_2

        m = (x_2 - x_1) / (y_2 - y_1)
        et.setdefault(y_min, []).append([y_max - 1, x_min, m, x_max])

    # initialising aet, preparing for loop
    aet = []
    y = min(et.keys())

    while aet or y in et.keys():
        # advancing scanline and updating aet accordingly
        aet = [edge for edge in aet if edge[0] >= y]
        for edge in aet:
            edge[1] += edge[2]
        if y in et:
            aet.extend(et.get(y))

        # sorting aet according to x-value
        intersections = sorted(edge[1] for edge in aet)

        # filling in odd pixels
        for i in range(0, len(intersections), 2):
            x_start = math.ceil(intersections[i])
            x_end = math.floor(intersections[i + 1])
            for x in range(x_start, x_end + 1):
                if 0 <= x < width and 0 <= y < height:
                    img[y, x] = fill_color
        y += 1

    # adding horizontal edges
    for edge in hor_edges:
        x_start, x_end = edge[0], edge[1]
        for x in range(x_start, x_end + 1):
            img[y, x] = fill_color

    # adding missing y_max
    for y_min in et.values():
        for edge in y_min:
            img[int(edge[0])+1, int(edge[3])] = fill_color
    return img

'''
#img = img_handler.grab_image("imgs//test_img_7.png")
#img_handler.display_img_array(scanline_queuebased_flood_fill_4con(img, (1, 0), 255))

img = img_handler.grab_image("imgs//test_img_6.png")
img_handler.display_img_array(scanline_stackbased_flood_fill_4con(img, (344, 190), 0))

img = img_handler.grab_image("imgs//test_img_6.png")
img_handler.display_img_array(scanline_stackbased_flood_fill_8con(img, (344, 190), 0))

img = img_handler.grab_image("imgs//test_img_6.png")
img_handler.display_img_array(scanline_stackbased_flood_fill_4con_optimized(img, (344, 190), 0))

img = img_handler.grab_image("imgs//test_img_6.png")
img_handler.display_img_array(scanline_stackbased_flood_fill_8con_optimized(img, (344, 190), 0))

faster_norm = 0
faster_opt = 0
for i in range(1000):
    print(i)

    img = img_handler.grab_image("imgs//test_img_6.png")
    start_1 = time.time()
    scanline_stackbased_flood_fill_4con(img, (344, 190), 0)
    end_1 = time.time()

    img = img_handler.grab_image("imgs//test_img_6.png")
    start_2 = time.time()
    scanline_stackbased_flood_fill_4con_optimized(img, (344, 190), 0)
    end_2 = time.time()

    if end_2 - start_2 <= end_1 - start_1:
        faster_opt +=1
    else:
        faster_norm +=1
print(faster_opt)
print(faster_norm)
'''