# -*- coding: utf-8 -*- 
# @Time : 2020/11/2 4:47 下午 
# @Author : yl

import cv2

import numpy as np

import mtcv

try:
    import torch
except ImportError:
    torch = None


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """Convert tensor to 3-channel images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W).
        mean (tuple[float], optional): Mean of images. Defaults to (0, 0, 0).
        std (tuple[float], optional): Standard deviation of images.
            Defaults to (1, 1, 1).
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and tensor.ndim == 4
    assert len(mean) == 3
    assert len(std) == 3

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mtcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE):
    return cv2.findContours(img, mode, method)