import torch.nn as nn
import torch
from .activation import Hard_sigmoid


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SpatialAttention(nn.Module):
    """Th spatial attention module in CBAM
    https://arxiv.org/pdf/1807.06521.pdf
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7)
        "kernel size must be 3 or 7"
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    """The channel attention module in CBAM.
    https://arxiv.org/pdf/1807.06521.pdf
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))

        out = avg_out + max_out
        return self.sigmoid(out)


class SqueezeExcite(nn.Module):
    """The channel attention module in Squeeze-and-excitation paper.
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels, ratio=0.25, reduced_base_channels=None,
                 act_layer=nn.ReLU, gate_fn=Hard_sigmoid, divisor=4):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn()
        reduced_channels = _make_divisible((reduced_base_channels or in_channels) * ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, reduced_channels, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_channels, in_channels, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x
