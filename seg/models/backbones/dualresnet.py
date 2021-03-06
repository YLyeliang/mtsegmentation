# -*- coding: utf-8 -*- 
# @Time : 2021/3/3 2:52 下午 
# @Author : yl

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
from ..builder import BACKBONES
from mtcv.runner import load_checkpoint
from seg.utils import get_root_logger
from mtcv.cnn import ConvModule, build_norm_layer

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False,
                 norm_cfg=dict(type='BN', momentum=bn_mom)):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear') + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


@BACKBONES.register_module()
class DualResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 base_channels=64,
                 spp_planes=128,
                 strides=[1, 2, 2, 2],
                 dilations=[1, 1, 1, 1],
                 bilateral_on_stage=[1, 1],
                 norm_cfg=dict(type='BN', momentum=bn_mom),
                 auxiliary=True):
        super(DualResNet, self).__init__()

        block, layers = self.arch_settings[depth]
        highres_planes = base_channels * 2
        self.auxiliary = auxiliary
        self.norm_cfg = norm_cfg
        self.bilateral_on_stage = bilateral_on_stage

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            build_norm_layer(norm_cfg, base_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            build_norm_layer(norm_cfg, base_channels)[1],
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=False)

        self.inplanes = base_channels
        self.layers = nn.ModuleList()
        self.compressions = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.layers_ = nn.ModuleList()
        for i, num_blocks in enumerate(layers):
            planes = base_channels * 2 ** i  # 64 128 256 512

            if i > 1:  # start from stage 3
                highres_inplanes = base_channels * 2

                # if bilateral_onstage[i-2]>1:
                for j in range(bilateral_on_stage[i - 2]):
                    # down-sample way
                    res_layer = self._make_layer(block, self.inplanes, planes,
                                                 num_blocks // bilateral_on_stage[i - 2],
                                                 stride=strides[i])
                    self.layers.append(res_layer)
                    # layers on high-resolution
                    layer_ = self._make_layer(block, highres_inplanes, highres_planes,
                                              num_blocks // bilateral_on_stage[i - 2])
                    highres_inplanes = highres_planes
                    self.layers_.append(layer_)

                    # bilateral up way
                    compression = nn.Sequential(nn.Conv2d(planes, highres_planes, kernel_size=1, bias=False),
                                                BatchNorm2d(highres_planes, momentum=bn_mom))
                    self.compressions.append(compression)
                    self.inplanes = planes * block.expansion
                    strides[i] = 1  # first one with down-sample, next without

                    # bilateral down way
                    down_inplanes = highres_planes
                    down_planes = planes // 2 if i - 1 == 2 else planes  # stage 4 have 2 down-sample layer with each 2x-down
                    down = nn.ModuleList()
                    for k in range(i - 1):  # start from stage 3(i==2), down-sample to 1/x from 1/8
                        if k == i - 2:  # which means no relu on final layer
                            conv_bn_act = ConvModule(down_inplanes, down_planes, kernel_size=3, stride=2, padding=1,
                                                     norm_cfg=norm_cfg, act_cfg=None)
                        else:
                            conv_bn_act = ConvModule(down_inplanes, down_planes, kernel_size=3, stride=2, padding=1,
                                                     norm_cfg=norm_cfg)
                        down.append(conv_bn_act)
                        down_inplanes = down_planes
                        down_planes += down_planes
                    down = nn.Sequential(*down)
                    self.downs.append(down)
            else:
                res_layer = self._make_layer(block, self.inplanes, planes, num_blocks, stride=strides[i])
                self.layers.append(res_layer)

            self.inplanes = planes * block.expansion

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)  # 128*2=256

        self.layer5 = self._make_layer(Bottleneck, planes, planes, 1, stride=2)  # 512*2=1024

        self.spp = DAPPM(planes * 2, spp_planes, planes // 2)  # 1024 128 256

        # if self.augment:
        #     self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)  # 128 128
        #
        # self.final_layer = segmenthead(planes * 4, head_planes, num_classes)  # 256

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1]
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # s 4
        auxiliary_layer = self.bilateral_on_stage[0] + 1
        outputs = []

        x = self.layers[0](x)  # s 4
        x = self.layers[1](self.relu(x))  # 8
        x_ = x
        x = self.layers[2](self.relu(x))  # 16
        x_later = x
        x_ = self.layers_[0](self.relu(x_))  # 8
        x = x + self.downs[0](self.relu(x_))  # 16
        width_output = x_.shape[-1]
        height_output = x_.shape[-2]
        x_ = x_ + F.interpolate(self.compressions[0](self.relu(x_later)),
                                size=[height_output, width_output],
                                mode='bilinear')
        if auxiliary_layer < 3: outputs.append(x_)  # ddr-23 append auxiliary layer now

        for i in range(3, len(self.layers)):
            x = self.layers[i](self.relu(x))
            x_later = x
            x_ = self.layers_[i - 2](self.relu(x_))
            x = x + self.downs[i - 2](self.relu(x_))
            x_ = x_ + F.interpolate(self.compressions[i - 2](self.relu(x_later)),
                                    size=[height_output, width_output],
                                    mode='bilinear')
            if auxiliary_layer == i:  # ddr-39 append auxiliary layer now
                outputs.append(x_)

        x_ = self.layer5_(self.relu(x_))  # 8
        x = F.interpolate(  # up(dappm(rbb(x))) # 64 -> 8
            self.spp(self.layer5(self.relu(x))),
            size=[height_output, width_output],
            mode='bilinear')

        output = self.relu(x + x_)

        if self.augment:
            return [temp, output]
        else:
            return [output]
