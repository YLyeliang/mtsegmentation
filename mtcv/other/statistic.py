# -*- coding: utf-8 -*- 
# @Time : 2021/3/8 2:34 下午 
# @Author : yl
import os
import matplotlib.pyplot as plt
import cv2


def aspectRatio(img_dir):
    """
    Statistic on aspect ratio of images
    Args:
        img_dir:

    Returns:

    """
    files = os.listdir(img_dir)
    # fig = plt.figure()
    x = []
    for file in files:
        img = os.path.join(img_dir, file)
        img = cv2.imread(img)
        h, w, _ = img.shape
        ratio = w / h
        x.append(ratio)
    y = [1] * len(x)
    plt.plot(x, y, 'yo-')
    plt.show()
