# 语义分割框架和模型

该框架基于mmsegmentation，在使用其基础框架的基础上进行开发。 其中，configs的写法、总体的框架体系`apis-datasets-models-ops-utils`和mmsegmentation保持一致，
详情可参考官方文档

相关变化如下：

- 使用本地编写的mtcv代替mmcv的库(可替换)
- 对相关训练、推理的API进行了修改

## 支持的模型和功能

### 模型

- [x] ddrnet(本项目使用模型为ddrnet-23)
- [x] deeplabv3
- [x] fcn
- [x] sfsegnet(实时语义分割模型)
- [x] u2net(显著性目标检测/分割模型)

### 功能

- [ ] 分布式训练（由于任务简单，2080ti任务单卡训练时间在10小时以内，未使用分布式训练）
- [x] pytorch-onnx-tensorrt(支持torch转onnx转tensorrt)，trt推理

## 训练流程

1. 在`configs/模型文件夹`下修改对应的配置文件。 比如`configs/ddrnet/ddrnet_23-d4_512x512.py`。 其中`configs/_base_`为通用继承的配置文件，包括数据集，模型和训练超参。
   通过修改`configs/_base_/datasets/paint.py`中的`data`参数，来修改训练过程中的训练样本图片和标签。 由于任务测试样本原因（无标签，较少），在训练阶段未使用测试集和验证集。

2. `python tools/train.py`进行训练，额外的命令行参数可参考训练文件，主要包括
   `--config 配置文件目录 --work-dir 模型保存地址 --gpu-ids 显卡编号(仅支持单卡）`

## tensorrt加速流程

首先使用`pytorch2onnx.py`将pytorch权重`pth`转为`onnx`格式。 然后使用`tensorrt`的linux二进制可执行命令接口`trtexec`对权重进行转化成二进制格式，项目中使用fp16进行量化。

## 一些实验记录

见实验日志：[research_log](research_log.md)
