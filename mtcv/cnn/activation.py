import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Hard_sigmoid(nn.Module):
    """A faster approximation of the sigmoid activation.
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.add_(3.).clamp_(0., 6.).div_(6.)
        else:
            return F.relu6(x + 3.) / 6.


act_cfg = {
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6,
    "LeakyReLU": nn.LeakyReLU,
    "Mish": Mish,
    "Hard_sigmoid": Hard_sigmoid
}


def build_act_layer(cfg, *args, **kwargs):
    """
    Build activation layer.
    Args:
        cfg(dict):cfg should contains:
        type(str): identify activation layer type

    Returns:

    """
    if cfg is None:
        cfg_ = dict(type="ReLU")
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in act_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        act_layer = act_cfg[layer_type]
    return act_layer(*args, **kwargs)
