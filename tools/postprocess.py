# -*- coding: utf-8 -*-
# @Time : 2020/11/23 10:36 上午 
# @Author : yl
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def morphologyClose(img, op=cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1):
    closed = cv2.morphologyEx(img, op=op, kernel=kernel, iterations=iterations)
    return closed


def morphologyOpen(img, op=cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1):
    opened = cv2.morphologyEx(img, op=op, kernel=kernel, iterations=iterations)
    return opened


def HoughLine(img, minLineLenggh=200, maxLineGap=15):
    lines = cv2.HoughLinesP(img, 10, np.pi / 180, 200)
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(img, (x1, y1), (x2, y2), 128, 3)


def areaRange(img, minArea=1000, maxArea=5000):
    """
    Judge the area of mask image whether in the range of the area.
    Args:
        img (ndarrays): gray image with shape (h,w)
        minArea (int):
        maxArea (int):

    Returns:

    """
    area = np.count_nonzero(img)
    h, w = img.shape
    maxArea = h * w // 10
    minArea = h * w // 300
    return True if minArea < area < maxArea else False
