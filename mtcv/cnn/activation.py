import torch
import torch.nn as nn


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


act_cfg = {
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6,
    "LeakyReLU": nn.LeakyReLU,
    "Mish": Mish
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
