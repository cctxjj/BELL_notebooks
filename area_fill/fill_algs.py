import numpy as np

import util.image_handler as imgutil

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
                stack.append((cur[1]+1, cur[0]))
        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] == start_col:
                stack.append((cur[1]-1, cur[0]))
        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                stack.append((cur[1], cur[0]+1))
        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                stack.append((cur[1], cur[0]-1))
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
                stack.append((cur[1]+1, cur[0]))
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] == start_col:
                    stack.append((cur[1]+1, cur[0] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] == start_col:
                    stack.append((cur[1] + 1, cur[0] - 1))

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] == start_col:
                stack.append((cur[1] - 1, cur[0]))
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] == start_col:
                    stack.append((cur[1] - 1, cur[0] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] == start_col:
                    stack.append((cur[1] - 1, cur[0] - 1))

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                stack.append((cur[1], cur[0] + 1))

        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                stack.append((cur[1], cur[0]-1))
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
                stack.append((cur[1]+1, cur[0]))
        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                stack.append((cur[1]-1, cur[0]))
        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                stack.append((cur[1], cur[0]+1))
        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                stack.append((cur[1], cur[0]-1))
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
                stack.append((cur[1]+1, cur[0]))
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] != boundary_color and img[cur[1] + 1, cur[0] + 1] != new_color:
                    stack.append((cur[1]+1, cur[0] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] != boundary_color and img[cur[1] + 1, cur[0] - 1] != new_color:
                    stack.append((cur[1] + 1, cur[0] - 1))

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                stack.append((cur[1] - 1, cur[0]))
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] != boundary_color and img[cur[1] - 1, cur[0] + 1] != new_color:
                    stack.append((cur[1] - 1, cur[0] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] != boundary_color and img[cur[1] - 1, cur[0] - 1] != new_color:
                    stack.append((cur[1] - 1, cur[0] - 1))

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                stack.append((cur[1], cur[0] + 1))

        if 0 <= cur[0] - 1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                stack.append((cur[1], cur[0]-1))
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
                queue.append((cur[1]+1, cur[0]))
        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] == start_col:
                queue.append((cur[1]-1, cur[0]))
        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                queue.append((cur[1], cur[0]+1))
        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                queue.append((cur[1], cur[0]-1))
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
                queue.append((cur[1]+1, cur[0]))
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] == start_col:
                    queue.append((cur[1]+1, cur[0] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] == start_col:
                    queue.append((cur[1] + 1, cur[0] - 1))

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] == start_col:
                queue.append((cur[1] - 1, cur[0]))
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] == start_col:
                    queue.append((cur[1] - 1, cur[0] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] == start_col:
                    queue.append((cur[1] - 1, cur[0] - 1))

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] == start_col:
                queue.append((cur[1], cur[0] + 1))

        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] == start_col:
                queue.append((cur[1], cur[0]-1))
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
                queue.append((cur[1]+1, cur[0]))
        if 0 <= cur[1]-1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                queue.append((cur[1]-1, cur[0]))
        if cur[0]+1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                queue.append((cur[1], cur[0]+1))
        if 0 <= cur[0]-1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                queue.append((cur[1], cur[0]-1))
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
                queue.append((cur[1]+1, cur[0]))
            if cur[0] + 1 < width:
                if img[cur[1] + 1, cur[0] + 1] != boundary_color and img[cur[1] + 1, cur[0] + 1] != new_color:
                    queue.append((cur[1]+1, cur[0] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] + 1, cur[0] - 1] != boundary_color and img[cur[1] + 1, cur[0] - 1] != new_color:
                    queue.append((cur[1] + 1, cur[0] - 1))

        if 0 <= cur[1] - 1:
            if img[cur[1] - 1, cur[0]] != boundary_color and img[cur[1] - 1, cur[0]] != new_color:
                queue.append((cur[1] - 1, cur[0]))
            if cur[0] + 1 < width:
                if img[cur[1] - 1, cur[0] + 1] != boundary_color and img[cur[1] - 1, cur[0] + 1] != new_color:
                    queue.append((cur[1] - 1, cur[0] + 1))
            if 0 <= cur[0] - 1:
                if img[cur[1] - 1, cur[0] - 1] != boundary_color and img[cur[1] - 1, cur[0] - 1] != new_color:
                    queue.append((cur[1] - 1, cur[0] - 1))

        if cur[0] + 1 < width:
            if img[cur[1], cur[0] + 1] != boundary_color and img[cur[1], cur[0] + 1] != new_color:
                queue.append((cur[1], cur[0] + 1))

        if 0 <= cur[0] - 1:
            if img[cur[1], cur[0] - 1] != boundary_color and img[cur[1], cur[0] - 1] != new_color:
                queue.append((cur[1], cur[0]-1))
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
        while img[y, left_b] == start_col and left_b > 0:
            img[y, left_b] = new_color
            left_b -= 1
        while img[y, right_b] == start_col and right_b < width-1:
            img[y, right_b] = new_color
            right_b += 1

    while stack:
        cur = stack.pop()
        identify_and_fill_run(cur)
        if cur[1] + 1 < height:
            if img[cur[1]+1, cur[0]] == start_col:
                stack.append((cur[0], cur[1] + 1))
        if cur[1] - 1 >= 0:
            if img[cur[1] - 1, cur[0]] == start_col:
                stack.append((cur[0], cur[1] - 1))
    return img
# Todo: check for side arms --> nochmal nachlesen

ar = imgutil.format_greyscale_img("imgs//test_img_2.png")
print(np.shape(ar))
imgutil.display_img_array(scanline_stackbased_flood_fill_4con(ar, (145, 266), 255))

