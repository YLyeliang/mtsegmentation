# -*- coding: utf-8 -*- 
# @Time : 2020/11/4 3:31 下午 
# @Author : yl

from .fcn_head import FCNHead
from .aspp_head import ASPPHead
from .u2net_head import U2NetHead

__all__ = [
    'FCNHead', 'ASPPHead', 'U2NetHead'
]
