# -*- coding: utf-8 -*- 
# @Time : 2020/11/18 3:44 下午 
# @Author : yl
import torch
import torch.nn as nn
import torch.nn.functional as F

from seg.ops import resize
from ..builder import BACKBONES

from mtcv.utils.parrots_wrapper import _BatchNorm
from mtcv.cnn import kaiming_init, constant_init
from mtcv.cnn import ConvModule


class RSU(nn.Module):
    """RSU block in U-2-Net

    """

    def __init__(self,
                 inplane,
                 mid_plane,
                 out_plane,
                 num_layers,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 ):
        super(RSU, self).__init__()

        self.inplane = inplane
        self.mid_plane = mid_plane
        self.out_plane = out_plane
        self.num_layers = num_layers

        self.conv = ConvModule(inplane, out_plane, kernel_size=3, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                               padding=1)
        self.enc_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(num_layers):
            # encode part
            if i == 0:
                enc_conv = ConvModule(out_plane, mid_plane, kernel_size=3, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                      act_cfg=act_cfg, padding=1)
            elif i == num_layers - 1:
                enc_conv = ConvModule(mid_plane, mid_plane, kernel_size=3, dilation=2, padding=2, conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg, act_cfg=act_cfg)
            else:
                enc_conv = ConvModule(mid_plane, mid_plane, kernel_size=3, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                      act_cfg=act_cfg, padding=1)
            if i < num_layers - 2:
                self.pools.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))

            # decode part
            if i > 0:
                if i == num_layers - 1:
                    dec_conv = ConvModule(mid_plane * 2, out_plane, kernel_size=3, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                          act_cfg=act_cfg, padding=1)
                else:
                    dec_conv = ConvModule(mid_plane * 2, mid_plane, kernel_size=3, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                          act_cfg=act_cfg, padding=1)
                self.dec_convs.append(dec_conv)
            self.enc_convs.append(enc_conv)

    def forward(self, x):
        identity = self.conv(x)
        enc_outs = []
        for i, enc in enumerate(self.enc_convs):
            x = enc(identity)
            if i < self.num_layers - 2:
                x = self.pools[i]
            enc_outs.append(x)
        for i in reversed(range(len(self.dec_convs))):
            if i == len(self.dec_convs) - 1:
                x = torch.cat([enc_outs[i + 1], enc_outs[i]], dim=1)
            else:
                x = torch.cat([enc_outs[i], x], dim=1)
            x = self.dec_convs[i](x)
            if i > 0:
                x = resize(x, size=enc_outs[i - 1], mode='bilinear')
        out = identity + x
        return out


class RSUF(nn.Module):
    """RSUF block in U-2-Net, which have no upsample layer and utilzied multiple dilated convs.

    """

    def __init__(self,
                 inplane,
                 mid_plane,
                 out_plane,
                 num_layers,
                 dilations=(1, 2, 4, 8),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 ):
        super(RSUF, self).__init__()

        self.inplane = inplane
        self.mid_plane = mid_plane
        self.out_plane = out_plane
        self.num_layers = num_layers

        self.conv = ConvModule(inplane, out_plane, kernel_size=3, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                               padding=1)
        self.enc_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(num_layers):
            # encode part
            if i == 0:
                enc_conv = ConvModule(out_plane, mid_plane, kernel_size=3, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                      act_cfg=act_cfg, padding=1)
            else:
                enc_conv = ConvModule(mid_plane, mid_plane, kernel_size=3, padding=dilations[i], dilation=dilations[i],
                                      conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

            # decode part
            if i > 0:
                if i == num_layers - 1:
                    dec_conv = ConvModule(mid_plane * 2, out_plane, kernel_size=3, padding=dilations[i],
                                          dilation=dilations[i], conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
                else:
                    dec_conv = ConvModule(mid_plane * 2, mid_plane, kernel_size=3, padding=dilations[i],
                                          dilation=dilations[i], conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
                self.dec_convs.append(dec_conv)
            self.enc_convs.append(enc_conv)

    def forward(self, x):
        identity = self.conv(x)
        enc_outs = []
        for i, enc in enumerate(self.enc_convs):
            x = enc(identity)
            enc_outs.append(x)
        for i in reversed(range(len(self.dec_convs))):
            if i == len(self.dec_convs) - 1:
                x = torch.cat([enc_outs[i + 1], enc_outs[i]], dim=1)
            else:
                x = torch.cat([enc_outs[i], x], dim=1)
            x = self.dec_convs[i](x)
            if i > 0 and self.upsample:
                x = resize(x, size=enc_outs[i - 1], mode='bilinear')
        out = identity + x
        return out


@BACKBONES.register_module()
class U2Net(nn.Module):
    """U-2-Net implementation

    """

    def __init__(self,
                 in_channels=3,
                 num_stages=5,
                 enc_num_layers=(7, 6, 5, 4, 4, 4),
                 dec_num_layers=(7, 6, 5, 4, 4),
                 enc_mid_channels=(32, 32, 64, 128, 256, 256),
                 dec_mid_channels=(16, 32, 64, 128, 256),
                 enc_out_channels=(64, 128, 256, 512, 512, 512),
                 dec_out_channels=(64, 64, 128, 256, 512),
                 out_indices=(0, 1, 2, 3, 4, 5),
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 dcn=None,
                 plugins=None):
        super(U2Net, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        assert len(enc_num_layers) == num_stages + 1, \
            'The length of enc_num_layers should be equal to num_stages, ' \
            f'while the enc_num_layers is {enc_num_layers}, the length of ' \
            f'enc_num_layers is {len(enc_num_layers)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(dec_num_layers) == num_stages, \
            'The length of dec_num_layers should be equal to (num_stages-1), ' \
            f'while the dec_num_layers is {dec_num_layers}, the length of ' \
            f'dec_num_layers is {len(dec_num_layers)}, and the num_stages is ' \
            f'{num_stages}.'

        self.num_stages = num_stages
        self.out_indices = out_indices
        self.align_corners = align_corners

        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.extra_layer = nn.ModuleList()
        extra_layer = len(enc_num_layers) > len(dec_num_layers)

        for i in range(num_stages):
            # encoder-decoder
            if not i == num_stages - 1:
                enc_conv_block = RSU(inplane=in_channels,
                                     mid_plane=enc_mid_channels[i],
                                     out_plane=enc_out_channels[i],
                                     num_layers=enc_num_layers[i],
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)
                dec_conv_block = RSU(inplane=enc_out_channels[i] * 2,
                                     mid_plane=enc_mid_channels[i],
                                     out_plane=dec_out_channels[i],
                                     num_layers=dec_num_layers[i],
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)
            else:
                enc_conv_block = RSUF(inplane=in_channels,
                                      mid_plane=dec_mid_channels[i],
                                      out_plane=enc_out_channels[i],
                                      num_layers=enc_num_layers[i],
                                      conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg)
                dec_conv_block = RSUF(inplane=enc_out_channels[i] * 2,
                                      mid_plane=dec_mid_channels[i],
                                      out_plane=dec_out_channels[i],
                                      num_layers=dec_num_layers[i],
                                      conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg)
            in_channels = enc_out_channels[i]
            self.encoder.append(enc_conv_block)
            self.pools.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))
            self.decoder.append(dec_conv_block)

        # extra layer
        if extra_layer:
            self.extra_layer.append(RSUF(inplane=in_channels,
                                         mid_plane=in_channels // 2,
                                         out_plane=enc_out_channels[-1],
                                         num_layers=enc_num_layers[-1], conv_cfg=conv_cfg,
                                         norm_cfg=norm_cfg,
                                         act_cfg=act_cfg))

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        assert pretrained is None, "pretrained model not supported in u2net."
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        # encoder
        enc_outs = []
        outs = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            enc_outs.append(x)
            x = self.pools[i](x)
        if self.extra_layer:
            x = self.encoder[-1](x)
            enc_outs.append(x)

        if self.out_indices[-1] >= self.num_stages:
            outs.append(x)

        # decoder
        for i in reversed(range(len(self.decoder))):
            x = resize(x, size=enc_outs[i].shape[2:], mode='bilinear', align_corners=self.align_corners)
            x = self.decoder[i](torch.cat([x, enc_outs[i]], dim=1))

            if i in self.out_indices:
                outs.append(x)
        outs.reverse()
        return outs