# -*- coding: utf-8 -*- 
# @Time : 2020/10/30 2:52 下午 
# @Author : yl

import cv2
import numpy as np
from mtcv.image.draw import draw_arrow, draw_circle, draw_ellipse
from mtcv.image import misc
from skimage.transform import PiecewiseAffineTransform, warp


def random_color():
    color = np.random.randint(0, 255, 3)
    return color.tolist()

    # def init_pen(thick,color):


def random_ellipse(img_h, img_w, return_axis=True):
    """Initialize the ellipse arguments with random location."""
    # center ,axes, angle, start angle, end angle
    draw_ellipse()


def random_circle(img_h, img_w, return_axis=True):
    """Initialize the circle arguments with random location.

    """
    radius_low = min(20, min(img_h, img_w) // 10)
    radius_high = max(20, min(img_h, img_w) // 5)
    thick_low = min(5, min(img_h, img_w) // 60)
    thick_high = max(6, min(img_h, img_w) // 30)

    thick = np.random.randint(thick_low, thick_high)
    radius = np.random.randint(radius_low, radius_high)
    x_coe = [0 + thick + radius, img_w - thick - radius]
    y_coe = [0 + thick + radius, img_h - thick - radius]
    center_x = np.random.randint(*x_coe)
    center_y = np.random.randint(*y_coe)

    contours = None
    if return_axis:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        draw_circle(mask, (center_x, center_y), radius, color=255, thick=1)
        contours, hierarchy = misc.findContours(mask)
        for i, contour in enumerate(contours):
            cnt_shape = contour.shape
            block = 10
            block_section = cnt_shape[0] // block
            rand = np.random.randint(-5, 10, [10, 1, 2])
            for j in range(10):
                contours[i][j * block_section:(j + 1) * block_section] += rand[j, ...]
        return (center_x, center_y), radius, thick, contours
        # debug:
        # tmp_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        # cv2.drawContours(tmp_mask, contours, 1, 128, thickness=thick)
        # cv2.imshow("tmp", tmp_mask)
        # cv2.waitKey()

    return (center_x, center_y), radius, thick


def random_arrow(img_h, img_w, minimum_length_scale=30, center=None, radius=None, return_axis=False):
    """Initialize the arrow arguments with random location.
    """
    thick_low = min(5, min(img_h, img_w) // 60)
    thick_high = max(6, min(img_h, img_w) // 30)

    thick = np.random.randint(thick_low, thick_high)

    circle_x, circle_y = center
    minimum_length = min(img_h, img_w) // minimum_length_scale

    if circle_x > img_w // 2:
        start_x = np.random.randint(0, circle_x - radius - minimum_length)
    else:
        start_x = np.random.randint(circle_x + radius + minimum_length, img_w)

    if circle_y > img_h // 2:
        start_y = np.random.randint(0, circle_y - radius - minimum_length)
    else:
        start_y = np.random.randint(circle_y + radius + minimum_length, img_h)

    # may positive or negative
    distance_x = circle_x - start_x - radius
    distance_y = circle_y - start_y - radius

    scale = np.random.uniform(0.5, 1)
    end_x = int(distance_x * scale) + start_x
    end_y = int(distance_y * scale) + start_y

    # TODO: add random warp on arrow
    if return_axis:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        draw_arrow(mask, (start_x, start_y), (end_x, end_y), color=255, thick=1)
        # because np.where return list of y and list of x. there needs some changes.
        idxs = np.asarray(np.where(mask > 0)).reshape(2, -1).transpose().reshape(-1, 1, 2)
        idxs = idxs[:, :, ::-1]
        # contours, hierarchy = misc.findContours(mask)
        # for i, contour in enumerate(contours):
        cnt_shape = idxs.shape
        block = 30
        block_section = cnt_shape[0] // block
        rand = np.random.randint(-5, 10, [block, 1, 2])
        for j in range(block):
            idxs[j * block_section:(j + 1) * block_section] += rand[j, ...]
        # debug:
        tmp_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        # cv2.drawContours(tmp_mask, contours, 0, 128, thickness=thick)
        cv2.polylines(tmp_mask, idxs, True, 255, thick)
        cv2.imshow("tmp", tmp_mask)
        cv2.waitKey()
        return (start_x, start_y), (end_x, end_y), thick, idxs

    return (start_x, start_y), (end_x, end_y), thick


def random_arrowv2(img_h, img_w, minimum_length_scale=20, center=None, radius=None, return_axis=False):
    """Initialize the arrow arguments with random location.
    """
    thick_low = min(5, min(img_h, img_w) // 60)
    thick_high = max(6, min(img_h, img_w) // 30)

    thick = np.random.randint(thick_low, thick_high)

    circle_x, circle_y = center
    minimum_length = min(img_h, img_w) // minimum_length_scale
    pixel_length_range = min(img_h, img_w) // minimum_length_scale
    if circle_x > img_w // 2:
        high = circle_x - radius - minimum_length
        low = max(0, high - pixel_length_range)
        start_x = np.random.randint(low, high)
    else:
        low = circle_x + radius + minimum_length
        high = min(img_w, low + pixel_length_range)
        start_x = np.random.randint(low, high)

    if circle_y > img_h // 2:
        high = circle_y - radius - minimum_length
        low = max(0, high - pixel_length_range)
        start_y = np.random.randint(low, high)
    else:
        low = circle_y + radius + minimum_length
        high = min(img_h, low + pixel_length_range)
        start_y = np.random.randint(low, high)

    # may positive or negative
    distance_x = circle_x - start_x - radius
    distance_y = circle_y - start_y - radius

    scale = np.random.uniform(0.5, 1)
    end_x = int(distance_x * scale) + start_x
    end_y = int(distance_y * scale) + start_y

    # TODO: add random warp on arrow
    if return_axis:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        draw_arrow(mask, (start_x, start_y), (end_x, end_y), color=255, thick=1)
        # because np.where return list of y and list of x. there needs some changes.
        idxs = np.asarray(np.where(mask > 0)).reshape(2, -1).transpose().reshape(-1, 1, 2)
        idxs = idxs[:, :, ::-1]
        # contours, hierarchy = misc.findContours(mask)
        # for i, contour in enumerate(contours):
        cnt_shape = idxs.shape
        block = 30
        block_section = cnt_shape[0] // block
        rand = np.random.randint(-5, 10, [block, 1, 2])
        for j in range(block):
            idxs[j * block_section:(j + 1) * block_section] += rand[j, ...]
        # debug:
        tmp_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        # cv2.drawContours(tmp_mask, contours, 0, 128, thickness=thick)
        cv2.polylines(tmp_mask, idxs, True, 255, thick)
        cv2.imshow("tmp", tmp_mask)
        cv2.waitKey()
        return (start_x, start_y), (end_x, end_y), thick, idxs

    return (start_x, start_y), (end_x, end_y), thick


def generate_arrow(start, end, direction=0):
    """Generate random arrow coordinates according to the start & end point, and direction.

    Args:
        start (tuple | list): Start point.
        end (tuple | list): End point.
        direction (int) : 0 means left convex, 1 means right convex.
    """
    assert len(start) == len(end) == 2
    start_x, start_y = start
    end_x, end_y = end
    if direction == 0:
        k = (end_y - start_y) / max((end_x - start_x), 1)  # the slop of the line

        b = end_y - k * end_x  # shift
        rand_num = max(end_y - start_y, end_x - start_x)  # the number of points used for draw curve.
        x = np.linspace(start_x, end_x, rand_num)
        # x = np.unique(x).astype(np.int8)
        y = np.linspace(start_y, end_y, rand_num)
        # y = np.unique(y).astype(np.int8)
        extreme = np.random.randint(rand_num // 3, rand_num - rand_num // 3)
        max_convex = np.random.randint(50)

        pass
    else:
        pass


# generate_arrow((50, 100), (200, 250))


def piecewiseAffineTrans(image):
    """
    Perform piecewise affine transformation on flags to fit the wave effect.
    Args:
        image: PIL image in mode RGBA.

    Returns:
        warped logo and corresponding point_list.
    """
    w, h = image.size

    cols_point = 20
    rows_point = 10
    wave_num = 1

    # choose the number of points in rows and cols. generate the meshgrid,
    src_cols = np.linspace(0, w, cols_point)
    src_rows = np.linspace(0, h, rows_point)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]  # (x,y)

    # add sinusoidal oscillation to row coordinates
    factor = np.random.randint(h // 15, h // 10)

    # rows +[0,3*pi], which decides the wave.
    dst_rows = src[:, 1] - np.sin(np.linspace(0, wave_num * np.pi, src.shape[0])) * factor
    dst_cols = src[:, 0]
    dst_rows *= 1.5
    dst_rows -= factor * 1.5
    dst = np.vstack([dst_cols, dst_rows]).T
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = int(h - 2 * factor)
    out_cols = w
    np_image = np.array(image)
    out = warp(np_image, tform, output_shape=(out_rows, out_cols), mode='constant', cval=0)
    out = out * 255
    out = out.astype(np.uint8)

    # image = Image.fromarray(out)

    return image
