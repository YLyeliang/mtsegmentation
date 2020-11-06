# -*- coding: utf-8 -*- 
# @Time : 2020/10/30 2:52 下午 
# @Author : yl

import cv2
import numpy as np
from mtcv.image.draw import draw_arrow, draw_circle
from mtcv.image import misc


def random_color():
    color = np.random.randint(0, 255, 3)
    return color.tolist()

    # def init_pen(thick,color):


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
