# -*- coding: utf-8 -*- 
# @Time : 2020/10/30 3:35 下午 
# @Author : yl

import cv2


def draw_rectangle(img, top_left, bottom_right, color=(0, 255, 0), thick=1):
    draw = cv2.rectangle(img, top_left, bottom_right, color, thickness=thick)
    return draw


def draw_circle(img, center, radius, color=(0, 255, 0), thick=1):
    draw = cv2.circle(img, center, radius, color, thickness=thick)
    return draw


def draw_arrow(img, start, end, color=(0, 255, 0), thick=1, line_type=None, tipLength=0.3):
    draw = cv2.arrowedLine(img, start, end, color, thickness=thick, line_type=line_type, tipLength=tipLength)
    return draw


def draw_line(img, start, end, color=(0, 255, 0), thick=1):
    draw = cv2.line(img, start, end, color, thickness=thick)
    return draw


def draw_ellipse(img, center, axes, angle, startAngle, endAngle, color=(0, 255, 0), thick=1):
    draw = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=thick)
    return draw


def draw_contour(img, contours, ct_id, color=(0, 255, 0), thick=1):
    draw = cv2.drawContours(img, contours, ct_id, color=color, thickness=thick)
    return draw


def polylines(img, points, isClosed=False, color=(0, 255, 0), thick=1):
    draw = cv2.polylines(img, points, isClosed, color, thickness=thick)
    return draw
