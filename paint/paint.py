# -*- coding: utf-8 -*- 
# @Time : 2020/10/30 2:53 下午 
# @Author : yl
import os
import cv2
import numpy as np
from mtcv.image.draw import draw_circle, draw_arrow, draw_contour
from .utils import random_color, random_circle, random_arrow
from mtcv.utils.misc import tictok
import os.path as osp
from mtcv import mkdir_or_exist


class DataPaint(object):
    """A painting data generator that paint random shape of circle on images.

    The whole process flow is as follows:
    1. get data root, and parse all image path into a list.
    2. for each image, initalize a pen with random color and thick.
    3. create a mask with same shape as raw image, paint random circle and arrow in the image.
    4. preserve the painted image and corresponding mask label.

    Args:
        data_root (str): The root dir of images waiting for paint.
        out_root (str): The out root dir of output images and annotation.s
        out_image_suffix (str): The output image dir.
        out_annot_suffix (str): The output annotation dir.
        class_id (dict): The class name to class_id.
        paint_loc (list): locations. useless now.
        paint_type (str): what kind of graph type painted on the image. Useless now.
        img_prefix (str): If not None, the data_root/img_prefix will be the image path.
    """

    def __init__(self,
                 data_root,
                 out_root,
                 out_image_suffix='image',
                 out_annot_suffix='annot',
                 img_suffix='jpg',
                 annot_suffix='png',
                 class_id={'background': 0, 'paint': 1},
                 paint_loc=[None],
                 paint_type=None,
                 img_prefix=None):
        self.data_root = data_root
        self.out_root = out_root
        self.out_image_suffix = out_image_suffix
        self.out_annot_suffix = out_annot_suffix
        self.img_suffix = img_suffix  # not used now
        self.annot_suffix = annot_suffix
        self.class_id = class_id
        self.paint_loc = paint_loc
        self.paint_type = paint_type
        self.img_prefix = img_prefix

    def paint(self, image, out_img=None, out_annot=None, color=None):
        """Paint arrow and circle on the images, and preserve painted image and corresponding annotations
        into specified path.

        Args:
            image (str): The abs path of image file.
            out_img (str): The abs path of output image file.
            out_annot (str): The abs path of output annotation file.

        Returns:

        """
        image = cv2.imread(image)
        if color is None:
            color = random_color()
        paint_id = self.class_id['paint']
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        center, radius, thick, contours = random_circle(h, w, return_axis=True)
        start, end, arrow_thick = random_arrow(h, w, center=center, radius=radius)
        circle = draw_contour(image, contours, 0, color=color, thick=thick)
        circle = draw_arrow(circle, start, end, color=color, thick=arrow_thick)
        circle_mask = draw_contour(mask, contours, 0, color=paint_id, thick=thick)
        circle_mask = draw_arrow(circle_mask, start, end, color=paint_id, thick=arrow_thick)
        if out_img:
            cv2.imwrite(out_img, circle)
        if out_annot:
            cv2.imwrite(out_annot, circle_mask)

    def check_path(self):
        """Check the validness of path, and make dirs.

        """
        if self.img_prefix:
            background_path = osp.join(self.data_root, self.img_prefix)
        else:
            background_path = self.data_root
        if not osp.exists(background_path): raise FileExistsError(
            f'image path {osp.abspath(background_path)} is not existed.')

        out_img_path = osp.join(self.out_root, self.out_image_suffix)
        out_annot_path = osp.join(self.out_root, self.out_annot_suffix)
        mkdir_or_exist(out_img_path)
        mkdir_or_exist(out_annot_path)

    def process(self):
        """Main process used to read background files, and paint some graphs on it.

        Returns:

        """
        timer = tictok()
        self.check_path()
        if self.img_prefix:
            background_path = osp.join(self.data_root, self.img_prefix)
        else:
            background_path = self.data_root
        files = os.listdir(background_path)

        out_img_path = osp.join(self.out_root, self.out_image_suffix)
        out_annot_path = osp.join(self.out_root, self.out_annot_suffix)
        print('Start generate data')
        for img_file in files:
            timer.tic()
            img = osp.join(background_path, img_file)
            out_img = osp.join(out_img_path, img_file)
            out_annot = osp.join(out_annot_path, img_file.split('.')[0] + f'.{self.annot_suffix}')
            self.paint(img, out_img, out_annot)
            timer.tok()
            print(f'process time cosuming: {timer.click:.2f}')


def paint(img, out_img=None, out_annot=None):
    image = cv2.imread(img)
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    circle = draw_circle(image, (30, 35), 30, thick=5)
    circle_mask = draw_circle(mask, (30, 35), 30, color=128, thick=5)
    draw_arrow()
    cv2.imshow('image', circle)
    cv2.imshow("mask", circle_mask)
    cv2.waitKey()


def rand_paint(img, out_img=None, out_annot=None):
    image = cv2.imread(img)
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center, radius, thick, contours = DataPaint.random_circle(h, w, return_axis=True)
    start, end, arrow_thick = DataPaint.random_arrow(h, w, center=center, radius=radius)
    # circle = draw_circle(image, center, radius, thick=thick)
    circle = draw_contour(image, contours, 0, thick=thick)
    circle = draw_arrow(circle, start, end, thick=arrow_thick)
    circle_mask = draw_contour(mask, contours, 0, color=128, thick=thick)
    circle_mask = draw_arrow(circle_mask, start, end, color=128, thick=arrow_thick)
    cv2.imshow('image', circle)
    cv2.imshow("mask", circle_mask)
    cv2.waitKey()

# file = "../images/005633.jpg"
# while True:
#     rand_paint(file)
