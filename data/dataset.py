"""
数据集加载和预处理
"""

import os
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision


def load_datasets(config, data_transforms):
    """
    加载数据集并创建数据加载器

    Args:
        config: 配置字典
        data_transforms: 数据转换字典

    Returns:
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        class_names: 类别名称列表
        class_to_idx: 类别到索引的映射
    """
    print("Loading datasets...")
    train_dir = os.path.join(config['dataset_path'], 'train')
    val_dir = os.path.join(config['dataset_path'], 'test')

    train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['Freeze_batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['Freeze_batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    config['num_train'] = len(train_dataset)
    config['num_val'] = len(val_dataset)
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    print(f"Classes: {class_names}")
    print(f"Class to index mapping: {class_to_idx}")
    print(f"Train samples: {config['num_train']}, Validation samples: {config['num_val']}")

    return train_loader, val_loader, train_dataset, val_dataset, class_names, class_to_idx


def visualize_dataset_samples(dataset, split_name, class_names, writer):
    """
    可视化数据集样本到TensorBoard

    Args:
        dataset: 数据集
        split_name: 数据集分割名称 (train/val)
        class_names: 类别名称列表
        writer: TensorBoard SummaryWriter
    """
    # 获取每个类别的索引
    class_indices = {}
    for i in range(len(dataset.targets)):
        target = dataset.targets[i]
        if target not in class_indices:
            class_indices[target] = []
        class_indices[target].append(i)

    # 为每个类别选择几个样本
    num_samples = min(5, min([len(indices) for indices in class_indices.values()]))

    for class_idx, indices in class_indices.items():
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        grid_images = []

        for idx in selected_indices:
            img, _ = dataset[idx]
            # 反归一化图像
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406]).view(3, 1, 1)
            grid_images.append(img)

        # 创建图像网格
        grid = torchvision.utils.make_grid(grid_images, nrow=num_samples)
        writer.add_image(f'Dataset/{split_name}_class_{class_names[class_idx]}', grid, 0)