# -*- coding: utf-8 -*- 
# @Time : 2020/10/30 2:52 下午 
# @Author : yl

import cv2
import numpy as np


def random_color():
    color = np.random.randint(0, 255, 3)
    return color

def init_pen(thick,color):

