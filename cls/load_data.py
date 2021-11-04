# -*- coding: utf-8 -*-
# @Time : 2021/10/18 下午2:18
# @Author: yl
# @File: load_data.py

from torchvision import transforms

import torch
import numpy as np

# torchvision version, which using PIL as backend.
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.485, .456, .406), (.229, .224, .225))
])

# cv2 version
