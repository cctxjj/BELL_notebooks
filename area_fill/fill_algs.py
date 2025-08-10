import numpy as np

import util.image_handler as imgutil
import time

def recursive_stackbased_flood_fill_4con(
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
    stack = [start]
    while stack:
        cur = stack.pop()
        img[cur[1], cur[0]] = new_color
        if cur[1]+1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1]+1))
        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1] - 1))
        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                stack.append((cur[0] + 1, cur[1]))
        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                stack.append((cur[0] - 1, cur[1]))
    return img

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
    print(height, width)
    start_col = img[start[1], start[0]]
    stack = [start]
    while stack:
        cur = stack.pop()
        img[cur[1], cur[0]] = new_color

        if cur[1] + 1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1]+1))
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] == start_col:
                    stack.append((cur[0] + 1, cur[1] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] == start_col:
                    stack.append((cur[0] - 1, cur[1] + 1))

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1] - 1))
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] == start_col:
                    stack.append((cur[0] + 1, cur[1] - 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] == start_col:
                    stack.append((cur[0] - 1, cur[1] - 1))

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                stack.append((cur[0] + 1, cur[1] ))

        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                stack.append((cur[0] - 1, cur[1]))
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
    while stack:
        cur = stack.pop()
        img[cur[1], cur[0]] = new_color

        if cur[1]+1 < height:
            if img[cur[1] + 1, cur[0]] != boundary_color and img[cur[1] + 1, cur[0]] != new_color:
                stack.append((cur[0], cur[1]+1))
        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                stack.append((cur[0], cur[1] - 1))
        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                stack.append((cur[0] + 1, cur[1]))
        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                stack.append((cur[0] - 1, cur[1]))
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
    print(height, width)
    stack = [start]
    while stack:
        cur = stack.pop()
        img[cur[1], cur[0]] = new_color

        if cur[1] + 1 < height:
            if img[cur[1] + 1, cur[0]] != boundary_color and img[cur[1] + 1, cur[0]] != new_color:
                stack.append((cur[0], cur[1]+1))
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] != boundary_color and img[cur[1] + 1, cur[0] + 1] != new_color:
                    stack.append((cur[0] + 1, cur[1] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] != boundary_color and img[cur[1] + 1, cur[0] - 1] != new_color:
                    stack.append((cur[0] - 1, cur[1] + 1))

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                stack.append((cur[0], cur[1] - 1))
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] != boundary_color and img[cur[1] - 1, cur[0] + 1] != new_color:
                    stack.append((cur[0] + 1, cur[1] - 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] != boundary_color and img[cur[1] - 1, cur[0] - 1] != new_color:
                    stack.append((cur[0] - 1, cur[1] - 1))

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                stack.append((cur[0] + 1, cur[1] ))

        if 0 <= cur[0] - 1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                stack.append((cur[0] - 1, cur[1]))
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
    queue = [start]
    while queue:
        cur = queue.pop(0)
        img[cur[1], cur[0]] = new_color

        if cur[1]+1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                queue.append((cur[0], cur[1] + 1))
        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] == start_col:
                queue.append((cur[0], cur[1] - 1))
        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                queue.append((cur[0] + 1, cur[1]))
        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                queue.append((cur[0] - 1, cur[1]))
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
    print(height, width)
    start_col = img[start[1], start[0]]
    queue = [start]
    while queue:
        cur = queue.pop(0)
        img[cur[1], cur[0]] = new_color

        if cur[1] + 1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                queue.append((cur[0], cur[1] + 1))
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] == start_col:
                    queue.append((cur[0] + 1, cur[1] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] == start_col:
                    queue.append((cur[0] - 1, cur[1] + 1))

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] == start_col:
                queue.append((cur[0], cur[1] - 1))
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] == start_col:
                    queue.append((cur[0] + 1, cur[1] - 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] == start_col:
                    queue.append((cur[0] - 1, cur[1] - 1))

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                queue.append((cur[0] + 1, cur[1]))

        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                queue.append((cur[0] - 1, cur[1]))
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
    queue = [start]
    while queue:
        cur = queue.pop(0)
        img[cur[1], cur[0]] = new_color

        if cur[1]+1 < height:
            if img[cur[1] + 1, cur[0]] != boundary_color and img[cur[1] + 1, cur[0]] != new_color:
                queue.append((cur[0], cur[1] + 1))
        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                queue.append((cur[0], cur[1] - 1))
        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                queue.append((cur[0] + 1, cur[1]))
        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                queue.append((cur[0] - 1, cur[1]))
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
    queue = [start]
    while queue:
        cur = queue.pop(0)
        img[cur[1], cur[0]] = new_color

        # noinspection DuplicatedCode
        if cur[1] + 1 < height:
            if img[cur[1] + 1, cur[0]] != boundary_color and img[cur[1] + 1, cur[0]] != new_color:
                queue.append((cur[0], cur[1] + 1))
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] != boundary_color and img[cur[1] + 1, cur[0] + 1] != new_color:
                    queue.append((cur[0] + 1, cur[1] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] != boundary_color and img[cur[1] + 1, cur[0] - 1] != new_color:
                    queue.append((cur[0] - 1, cur[1] + 1))

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                queue.append((cur[0], cur[1] - 1))
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] != boundary_color and img[cur[1] - 1, cur[0] + 1] != new_color:
                    queue.append((cur[0] + 1, cur[1] - 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] != boundary_color and img[cur[1] - 1, cur[0] - 1] != new_color:
                    queue.append((cur[0] - 1, cur[1] - 1))

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                queue.append((cur[0] + 1, cur[1]))

        if 0 <= cur[0] - 1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                queue.append((cur[0] - 1, cur[1]))
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
    start_col = img[start[1], start[0]] # switched weil kp

    stack = [start]

    def identify_and_fill_run(seed):
        img[seed[1], seed[0]] = new_color
        left_b, y = seed[0]-1, seed[1]
        right_b = left_b+2

        run_above_detected_on_start = False
        run_below_detected_on_start = False
        if y + 1 < height:
            if img[cur[1] + 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1] + 1))
                run_above_detected_on_start = True
        if y - 1 >= 0:
            if img[cur[1] - 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1] - 1))
                run_below_detected_on_start = True

        # looking for the left boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while img[y, left_b] == start_col and left_b > 0:
            img[y, left_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, left_b] != start_col:
                        run_above = False
                elif img[y + 1, left_b] == start_col:
                    run_above = True
                    stack.append((left_b, y + 1))

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, left_b] != start_col:
                        run_below = False
                elif img[y - 1, left_b] == start_col:
                    run_below = True
                    stack.append((left_b, y - 1))
            left_b -= 1

        # looking for the right boundary, scanning for new runs above/below
        run_above = run_above_detected_on_start
        run_below = run_below_detected_on_start
        while img[y, right_b] == start_col and right_b < width-1:
            img[y, right_b] = new_color

            if y + 1 < height:
                if run_above:
                    if img[y + 1, right_b] != start_col:
                        run_above = False
                elif img[y + 1, right_b] == start_col:
                    run_above = True
                    stack.append((right_b, y + 1))

            if y - 1 >= 0:
                if run_below:
                    if img[y - 1, right_b] != start_col:
                        run_below = False
                elif img[y - 1, right_b] == start_col:
                    run_below = True
                    stack.append((right_b ,y - 1))

            right_b += 1

    while stack:
        cur = stack.pop()
        identify_and_fill_run(cur)

    return img
