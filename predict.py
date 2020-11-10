# -*- coding: utf-8 -*- 
# @Time : 2020/11/6 10:13 上午 
# @Author : yl
from seg.apis import inference_segmentor, init_segmentor
import mtcv
import os.path as osp
import numpy as np

config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# or img = mmcv.imread(img), which will only load it once
img_dir = ""
out_dir = ""
img_list = mtcv.scandir(img_dir)

for img_name in img_list:
    img = osp.join(img_dir, img_name)
    result = inference_segmentor(model, img)
    # save the visualization results to image files, the result will be draw on raw image.
    out = osp.join(out_dir, img_name)
    model.show_result(img, result, out_file=out)

    # only draw results on mask
    seg = result[0]
    # mask = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8)
    classes = model.CLASSES
    mask = np.where(seg == 1, 255, 0)
    mtcv.imwrite(mask, out)
