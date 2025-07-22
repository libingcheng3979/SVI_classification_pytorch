"""
数据增强和变换
"""

import torch
from torchvision import transforms


def get_data_transforms(input_shape):
    """
    获取数据转换

    Args:
        input_shape: 输入图像的尺寸 [height, width]

    Returns:
        data_transforms: 训练集和验证集的数据转换字典
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_shape),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    return data_transforms