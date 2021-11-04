# -*- coding: utf-8 -*-
# @Time : 2021/10/18 下午1:55
# @Author: yl
# @File: train.py

import os
import sys

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

from .models.mobilev3 import mobilenet_v3_large
from .dataset import ImageFolder
from .load_data import train_transforms
from cls.models.resnet import resnet50
from PIL import Image
import cv2


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--data_prefix', default="/work/data/xx", help="root path of training set")
    args.add_argument('--start_iter', default=0)
    args.add_argument('--learning_rate', default=1e-3)
    args.add_argument('--max_epoch', default=150, type=int)
    args.add_argument('--num_workers', default=8)
    args.add_argument('--batch_size', default=128)
    args.add_argument('--gpu_id', default=4, type=int)
    args.add_argument('--in_channel', default=1, type=int)
    args.add_argument('--save_folder', default='./work-dirs/xx')
    args.add_argument('--load_from', type=str, default=None)
    args.add_argument('--max_keep', default=2, type=int)
    args.add_argument('--model', choices=['resnet', 'mobilev3'], default='mobilev3')
    args.add_argument('--use_sigmoid', default=False, type=str2bool)
    return args.parse_args()


def str2bool(str):
    if "True" in str:
        return True
    else:
        return False


if __name__ == '__main__':
    args = parse_args()
    print(f"Using sigmoid: {args.use_sigmoid}")
    dataset = ImageFolder(args.data_prefix, transform=train_transforms)

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

    if args.use_sigmoid:  # without negative class
        classes = len(dataset.classes) - 1
    else:
        classes = len(dataset.classes)
    if args.model == 'mobilev3':
        model = mobilenet_v3_large(pretrained="path", num_classes=classes, in_channel=args.in_channel)
    else:
        model = resnet50(True, checkpoint="xx", num_classes=classes, in_channel=args.in_channel)

    if args.load_from:
        checkpoint = torch.load(args.load_from, map_location='cpu')
        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    model = torch.nn.DataParallel(model, device_ids=[args.gpu_id])

    device = f"cuda:{args.gpu_id}"
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=0, last_epoch=-1)
    if args.use_sigmoid:
        loss_func = nn.BCEWithLogitsLoss()
    else:
        loss_func = nn.CrossEntropyLoss()

    max_batch = len(dataset) // args.batch_size
    model.train()
    meta = dict(CLASSES=dataset.classes)
    for epoch in range(1, args.max_epoch):
        s = time.time()
        num = 0
        batch = 0
        average_loss = 0.
        train_correct = 0.
        for batch_images, batch_labels in train_dataloader:
            if torch.cuda.is_available():
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

            num = len(batch_images)

            out = model(batch_images)
            if args.use_sigmoid:
                negative_idx = batch_labels == 0  # negative index: 0
                batch_labels = batch_labels - 1
                labels = batch_labels.clamp(min=0, max=classes - 1)
                labels = F.one_hot(labels, num_classes=classes).float()
                labels[negative_idx] = torch.zeros_like(labels)[0]  # negative, one-hot: 0 0 0 0
                loss = loss_func(out, labels)
                prediction = out.sigmoid()
                m, m_idx = prediction.max(1)
                m_valid = m > 0.5
                pred_correct = batch_labels.new_full(batch_labels.shape, -1)
                pred_correct[m_valid] = m_idx[m_valid]
                train_correct = pred_correct.eq(batch_labels).sum().item()
            else:
                loss = loss_func(out, batch_labels)
                prediction = torch.max(out, 1)[1]
                train_correct = prediction.eq(batch_labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch += 1
            average_loss = loss.item()

            if batch % 10 == 0 or batch == max_batch:
                end = time.time()
                print(f"epoch: {epoch}, batch: {batch}/{max_batch}, avg loss: {average_loss:.4f}, "
                      f"learning_rate: {scheduler.get_last_lr()[0]:e}, train_acc: {train_correct / num:.4f}, cost: {end - s:.1f} sec")
                s = time.time()

        scheduler.step()
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.max_epoch:
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)
            checkpoint = {
                'meta': meta,
                'state_dict': model.state_dict()
            }
            pth_name = f"epoch_{epoch + 1}.pth"
            torch.save(checkpoint, os.path.join(args.save_folder, pth_name))
            print(f"save model: {pth_name}")
            if args.max_keep:
                files = os.listdir(args.save_folder)
                ckpts = [_ for _ in files if "epoch" in _]
                ckpts.sort(key=lambda x: int(x.split('.')[0][6:]))
                if len(ckpts) > args.max_keep:
                    os.remove(os.path.join(args.save_folder, ckpts[0]))
