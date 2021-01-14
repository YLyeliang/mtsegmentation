# -*- coding: utf-8 -*- 
# @Time : 2020/11/5 10:19 上午
# @Author : yl
import torch
import torch.nn as nn
from mtcv.cnn import build_conv_layer

from seg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..losses import accuracy


@HEADS.register_module()
class U2NetHead(BaseDecodeHead):
    """U-2-Net for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_sides=6,
                 kernel_size=3,
                 use_sigmoid=True,
                 **kwargs):
        super(U2NetHead, self).__init__(**kwargs)
        self.num_sides = num_sides
        self.kernel_size = 3

        self.side_convs = nn.ModuleList()
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            assert self.num_classes == 1, 'sigmoid mode only useful for binary classes segmentation'

        for i in range(num_sides):
            side_conv = build_conv_layer(self.conv_cfg, in_channels=self.in_channels[i], out_channels=self.num_classes,
                                         kernel_size=kernel_size)
            self.side_convs.append(side_conv)

        if self.use_sigmoid:
            self.conv_seg = nn.Conv2d(self.num_sides, self.num_classes, kernel_size=1)
        else:
            self.conv_seg = nn.Conv2d(self.num_classes * self.num_sides, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        """Forward process."""
        x = self._transform_inputs(inputs)
        sides_out = []
        for i in range(self.num_sides):
            side = self.side_convs[i](x[i])
            side = resize(side, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            sides_out.append(side)

        output = torch.cat(sides_out, 1)
        output = self.cls_seg(output)
        sides_out.insert(0, output)
        return sides_out

    def losses(self, seg_logits, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        if self.use_sigmoid:
            if self.num_sides > 1:
                for i, seg_logit in enumerate(seg_logits):
                    batch = seg_logit.shape[0]
                    seg_label = seg_label.view(batch, -1)
                    loss[f'loss_seg_{i}'] = self.loss_decode(
                        seg_logit.sigmoid().view(batch, -1),
                        seg_label,
                        weight=seg_weight)
            else:
                batch = seg_logits.shape[0]
                seg_label = seg_label.view(batch, -1)
                loss[f'loss_seg'] = self.loss_decode(
                    seg_logits.sigmoid().view(batch, -1),
                    seg_label,
                    weight=seg_weight)
        else:
            if self.num_sides > 1:
                for i, seg_logit in enumerate(seg_logits):
                    seg_label = seg_label.squeeze(1)
                    loss[f'loss_seg_{i}'] = self.loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                    )
            else:
                seg_label = seg_label.squeeze(1)
                loss[f'loss_seg'] = self.loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight)

                # loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
