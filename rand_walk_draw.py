import numpy as np
import cv2

# # BGR
# canvas_size = (400, 400, 3)
# canvas_color = (255, 255, 255)
# draw_color = (0, 0, 0)

# GRAYSCALE
canvas_size = (400, 400)
canvas_color = 255
draw_color = 0

canvas = np.ones(canvas_size) * canvas_color
canvas = canvas.astype('uint8')


def weight_step(canvas, canvas_color, location, lo_weight=0.3):
    """Weight random walk to step towards the most blank areas of the canvas

    :param canvas:
    :param canvas_color:
    :param location:
    :param lo_weight:
    :return:
    """
    loc_y, loc_x = location

    left_of_x = canvas[:, :loc_x]
    right_of_x = canvas[:, loc_x:]
    above_y = canvas[:loc_y, :]
    below_y = canvas[loc_y:, :]

    n_blank_to_left = np.sum(left_of_x == canvas_color)
    n_blank_to_right = np.sum(right_of_x == canvas_color)
    n_blank_above = np.sum(above_y == canvas_color)
    n_blank_below = np.sum(below_y == canvas_color)

    hi_weight = 1 - lo_weight * 2
    if n_blank_to_right > n_blank_to_left:
        px = [lo_weight, lo_weight, hi_weight]
    else:
        px = [hi_weight, lo_weight, lo_weight]

    if n_blank_below > n_blank_above:
        py = [lo_weight, lo_weight, hi_weight]
    else:
        py = [hi_weight, lo_weight, lo_weight]

    return px, py


def random_step(canvas, canvas_color, location, max_tries=32):
    """Take a one pixel step to a blank pixel

    Blank is interpreted to be where canvas == canvas_color

    :param canvas: np.array that is the drawing canvas
    :param canvas_color: starting color of canvas
    :param location: (y, x) location where to take step from
    :param max_tries: how many failed step attempts before giving up;
                      failures can happen when walk ended up in a pixel surrounded
                      by non-blank pixels
    :return: a random, blank (y, x) location that is one step away from start loc
             this will be None if unable to take step
    """
    step_options = [-1, 0, 1]
    px, py = weight_step(canvas, canvas_color, location)

    new_loc = None

    max_y, max_x = canvas.shape[:2]
    loc_y, loc_x = location
    for _ in range(max_tries):
        step_x = np.random.choice(step_options, p=px)
        step_y = np.random.choice(step_options, p=py)

        new_x = loc_x + step_x
        new_y = loc_y + step_y

        if new_x < 0 or new_y < 0:
            continue

        if new_x > max_x - 1 or new_y > max_y - 1:
            continue

        possible_loc = new_y, new_x
        loc_color = canvas[possible_loc]

        if np.all(loc_color == canvas_color):
            new_loc = possible_loc
            break

    return new_loc


def start_loc(canvas, canvas_color):
    """Find a random location on canvas that is blank

    Blank is interpreted to be where canvas == canvas_color

    :param canvas: np.array that is the drawing canvas
    :param canvas_color: starting color of canvas
    :return: a random (y, x) location on canvas that is blank
    """
    blank_loc_y, blank_loc_x = np.where(canvas == canvas_color)[:2]
    rand_idx = np.random.choice(len(blank_loc_y))
    loc = blank_loc_y[rand_idx], blank_loc_x[rand_idx]

    return loc


if __name__ == '__main__':
    # Modified random walk:
    #   * Don't draw where not blank
    #   * Weight randomness towards most blank areas
    #   * Start in upper right
    #   * If can't randomly take step to a blank pixel; randomly start somewhere else
    n_starts = 100
    locs = [start_loc(canvas, canvas_color) for _ in range(n_starts)]
    locs = list(set(locs))

    while True:
        cv2.imshow('Art... prolly', canvas)
        cv2.waitKey(5)

        new_locs = []
        for i, loc in enumerate(locs):
            draw_color_i = draw_color

            canvas[loc] = draw_color_i

            loc = random_step(canvas, canvas_color, loc)
            if loc is not None:
                new_locs.append(loc)

        locs = new_locs

        if not locs:
            break

    cv2.destroyAllWindows()
    cv2.imshow('I give up...', canvas)
    cv2.waitKey(0)
