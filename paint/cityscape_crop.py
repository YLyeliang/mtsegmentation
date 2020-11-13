# -*- coding: utf-8 -*- 
# @Time : 2020/11/12 5:06 下午 
# @Author : yl
import os
import os.path as osp
from pathlib import Path
from PIL import Image


def crop(path, out_path):
    # dirs = os.listdir(path)
    dirs = Path(path)

    for dir in dirs.iterdir():
        if dir.is_dir():
            files = os.listdir(dir)
            for file in files:
                img_name = osp.join(dir, file)
                basename, suffix = img_name.split('.')
                img = Image.open(img_name)
                crop_size = (1024, 1024)
                crop1 = img.crop((0, 0, crop_size[0], crop_size[1]))
                crop2 = img.crop((crop_size[0], 0, 2048, 1024))
                crop1.save(osp.join(out_path, basename + '_1' + suffix))
                crop2.save(osp.join(out_path, basename + '_2' + suffix))


path = '/Users/fiberhome/Downloads/BaiduDisk/cityscape/val/'

crop(path, None)
