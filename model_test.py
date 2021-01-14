# -*- coding: utf-8 -*- 
# @Time : 2020/11/9 4:10 下午 
# @Author : yl
import torch
model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=False)
model.eval()
from torchvision import models
models.mobilenet_v2()