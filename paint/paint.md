# 自动化数据生成
该目录下包含画圈的自动化数据生成代码，代码主体为`paint.py`(生成过程见代码注释)。

流程简介：
- 使用真实场景的图片（日常拍摄、街景、户外），生成相同大小的mask图，在原图上随机生成圆圈和箭头（颜色+透明度），
同时对mask上对应像素位置的值修改为1，生成训练标签（0为背景，1为标签）

## 自动化数据生成流程
1. 下载cityscape数据集（可采用其他真实场景图片），其图片大小为2048*1024，这里仅使用原图
2. 使用cityscape_crop.py对图片进行裁剪
3. 修改工程目录/data_gen.py中的对应文件夹目录，运行得到训练集图片
