# -*- coding: utf-8 -*- 
# @Time : 2020/11/24 8:31 下午 
# @Author : yl

"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import math

from mtcv.cnn import build_conv_layer, build_norm_layer, SqueezeExcite, _make_divisible
from mtcv.cnn import kaiming_init, constant_init
from mtcv.utils.parrots_wrapper import _BatchNorm


class GhostModule(nn.Module):
    """Ghost module descbried in https://arxiv.org/abs/1911.11907.
    outpalne: n .  i
    ntrinsic feature maps: m.
    ghost features: s
    m = n/s.
    1. perform common conv to get m channels.
    2. perform cheap operation ( which is depth-wise conv) on each channel
    3. perform identity mapping on output feature maps (concat output of common conv & cheap op)
    The output channel is m+s
    """

    def __init__(self,
                 inplane,
                 outplane,
                 kernel_size=1,
                 s=2,
                 dw_size=3,
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_relu=True):
        super(GhostModule, self).__init__()
        self.outplane = outplane
        init_channels = math.ceil(outplane / s)
        new_channels = init_channels * (s - 1)

        self.primary_conv = nn.Sequential(
            build_conv_layer(conv_cfg,
                             inplane,
                             init_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=kernel_size // 2,
                             bias=False),
            build_norm_layer(norm_cfg, init_channels)[1],
            nn.ReLU(inplace=True) if with_relu else nn.Sequential())

        self.cheap_operation = nn.Sequential(
            build_conv_layer(conv_cfg,
                             init_channels,
                             new_channels,
                             kernel_size=dw_size,
                             stride=1,
                             padding=dw_size // 2 if dilation == 1 else dilation,
                             dilation=dilation,
                             groups=init_channels,
                             bias=False),
            build_norm_layer(norm_cfg, new_channels)[1],
            nn.ReLU(inplace=True) if with_relu else nn.Sequential())

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.outplane, :, :]


class GhostBottleneck(nn.Module):
    """Ghost bottleneck w/ optinal SE"""

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 dw_kernel_size=3,
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        with_se = se_ratio is not None and se_ratio > 0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_channels,
                                  mid_channels,
                                  with_relu=True)

        # Depth-wise conv
        if self.stride > 1:
            self.conv_dw = build_conv_layer(conv_cfg,
                                            mid_channels,
                                            mid_channels,
                                            kernel_size=dw_kernel_size,
                                            stride=stride,
                                            padding=(dw_kernel_size - 1) // 2 if dilation == 1 else dw_kernel_size // 2 * dilation,
                                            dilation=dilation,
                                            groups=mid_channels,
                                            bias=False)
            self.bn_dw = build_norm_layer(norm_cfg, mid_channels)[1]

        # Squeeze-and-excitation
        if with_se:
            self.se = SqueezeExcite(mid_channels, ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_channels, out_channels, with_relu=False)

        # shortcut
        if not (in_channels == out_channels and self.stride == 1):
            self.shortcut = nn.Sequential(
                build_conv_layer(conv_cfg,
                                 in_channels,
                                 in_channels,
                                 kernel_size=dw_kernel_size,
                                 stride=stride,
                                 padding=(dw_kernel_size - 1) // 2 if dilation == 1 else dw_kernel_size // 2 * dilation,
                                 dilation=dilation,
                                 groups=in_channels,
                                 bias=False),
                build_norm_layer(norm_cfg, in_channels)[1],
                build_conv_layer(conv_cfg,
                                 in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        if self.shortcut:
            x += self.shortcut(residual)
        else:
            x += residual
        return x


class GhostNet(nn.Module):
    """GhostNet: More features from cheap operations, https://arxiv.org/abs/1911.11907.

    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2],
         [3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2],
         [5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2],
         [3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]],
        # stage5
        [[5, 672, 160, 0.25, 2],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]]
    ]

    def __init__(self,
                 in_channels,
                 width=1.0,
                 dilations=(1, 1, 1, 1, 1),
                 strides=(1, 2, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 num_stages=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 ):
        super(GhostNet, self).__init__()
        self.in_channels = in_channels
        self.width = width
        self.out_incides = out_indices
        self.num_stages = num_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # build stem layer
        out_channel = _make_divisible(16 * width, 4)
        self.stem_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        in_channels = out_channel

        # building inverted residual blocks
        self.ghost_stages = nn.ModuleList()
        block = GhostBottleneck

        for i in range(self.num_stages):
            layers = []
            for kernel, exp_size, c, se_ratio, s in self.cfgs[i]:
                if s > strides[i]: s = strides[i]
                out_channel = _make_divisible(c * width, 4)
                hid_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(in_channels=in_channels,
                                    mid_channels=hid_channel,
                                    out_channels=out_channel,
                                    dw_kernel_size=kernel,
                                    stride=s,
                                    dilation=dilations[i],
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg,
                                    se_ratio=se_ratio))
                in_channels = out_channel
            self.ghost_stages.append(nn.Sequential(*layers))

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        assert pretrained is None, 'Pretrained model is not supported now'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        x = self.stem_layer(x)
        outs = []
        for i, stage in enumerate(self.ghost_stages):
            x = stage(x)
            if i in self.out_incides:
                outs.append(x)

        return outs
