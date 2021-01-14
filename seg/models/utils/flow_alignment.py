# -*- coding: utf-8 -*- 
# @Time : 2021/1/12 10:42 上午 
# @Author : yl
import torch
from torch import nn
import torch.nn.functional as F





class AlignedModule(nn.Module):
    """
    Perform semantic flow warp. https://arxiv.org/abs/2002.10120
    """

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        """
        Args:
            x (list): list of feature maps [low-level, high-level]
        """
        low_feature, h_feature = x
        h_feature_origin = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode='bilinear', align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_origin, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        """
        Args:
            input:
            flow:
            size:
        """
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)  # shape [1,1,1,2]
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)  # (out_h,out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)  # (out_h, out_w)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)  # (out_h, out_w, 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)  # (n, out_h, out_w, offset=2)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output
