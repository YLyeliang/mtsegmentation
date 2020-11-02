# -*- coding: utf-8 -*- 
# @Time : 2020/10/30 2:53 下午 
# @Author : yl


class DataPaint(object):
    """A painting data generator that paint random shape of circle on images.

    The whole process flow is as follows:
    1. get data root, and parse all image path into a list.
    2. for each image, initalize a pen with random color and thick.
    3. create a mask with same shape as raw image, paint random circle and arrow in the image.
    4. preserve the painted image and corresponding mask label.

    """

    def __init__(self,
                 data_root,
                 paint_loc=[None],
                 paint_type=None,
                 img_prefix=None, ):
        self.data_root = data_root
        self.paint_loc = paint_loc
        self.paint_type = paint_type
        self.img_prefix = img_prefix
