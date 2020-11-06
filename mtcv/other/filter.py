# -*- coding: utf-8 -*- 
# @Time : 2020/11/4 4:45 下午 
# @Author : yl

import os
import os.path as osp
import shutil
from PIL import Image


def aspect_ratio_filter(in_path, out_path, ratio=[0.2, 1.5]):
    """Filter images in the in_path according the specified aspect ratio.

    Args:
        in_path:
        out_path:
        ratio: images with aspect ratio in the given range will be preserved.

    Returns:

    """
    assert osp.exists(in_path) == True, f"Path {in_path} is not existed."
    files = os.listdir(in_path)
    files_path = [osp.join(in_path, file) for file in files]
    for img_path in files_path:
        img = Image.open(img_path)
        w, h = img.size
        aspect_ratio = w / h
        if aspect_ratio >= ratio[0] and aspect_ratio <= ratio[1]:
            shutil.copy(img_path, osp.join(out_path, osp.basename(img_path)))
        else:
            continue
