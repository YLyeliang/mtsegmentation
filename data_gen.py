# -*- coding: utf-8 -*- 
# @Time : 2020/11/3 4:28 下午 
# @Author : yl

from paint import DataPaint

data_root = '/Users/fiberhome/Downloads/data_road/testing/image_2'
out_root = '/Users/fiberhome/Downloads/paint_data/'

generator = DataPaint(data_root, out_root)

generator.process()
