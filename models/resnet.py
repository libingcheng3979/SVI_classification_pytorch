"""
模型构建函数
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(config, device):
    """
    构建模型

    Args:
        config: 配置字典
        device: 设备 (cuda/cpu)

    Returns:
        model: 构建好的模型
    """
    print("Building model...")
    # 首先创建没有预训练权重的模型
    model = models.resnet50(weights=None)

    # 加载Places365预训练权重
    print(f"Loading Places365 weights from {config['model_path']}...")
    try:
        # 加载Places365权重
        checkpoint = torch.load(config['model_path'])

        # 处理state_dict，去掉'module.'前缀(如果有)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}

        # 创建新的state_dict，排除fc层的权重(最后一层)
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'fc' not in k:  # 只保留非fc层的权重
                new_state_dict[k] = v

        # 加载除fc层外的所有层
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"Places365 weights loaded successfully (except FC layer)")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    except Exception as e:
        print(f"Error loading Places365 weights: {e}")
        print("Continuing with ImageNet pretrained weights...")
        model = models.resnet50(pretrained=True)

    # 修改全连接层以适应我们的类别数
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, config['num_classes'])
    )

    # 将模型移动到设备
    model = model.to(device)

    return model


def setup_optimizer(model, config, is_freeze=True):
    """
    设置优化器和学习率调度器

    Args:
        model: 模型
        config: 配置字典
        is_freeze: 是否冻结特征提取层

    Returns:
        optimizer: 优化器
        scheduler: 学习率调度器
        current_batch_size: 当前批次大小
        optimizer_info: 优化器信息
        scheduler_info: 学习率调度器信息
    """
    if is_freeze:
        # 冻结特征提取层
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        params = model.fc.parameters()
        base_lr = config['Init_lr']
        current_batch_size = config['Freeze_batch_size']
    else:
        # 解冻全部层，但使用不同的学习率
        for name, param in model.named_parameters():
            param.requires_grad = True

        # 底层使用较小的学习率，顶层使用较大的学习率
        params = [
            {'params': [p for n, p in model.named_parameters() if 'fc' not in n],
             'lr': config['Init_lr'] * 0.1},
            {'params': model.fc.parameters()}
        ]
        base_lr = config['Init_lr']
        current_batch_size = config['Unfreeze_batch_size']

    # 创建优化器
    if config['optimizer_type'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=config['momentum'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        optimizer_info = f"SGD(lr={base_lr}, momentum={config['momentum']}, weight_decay={config.get('weight_decay', 1e-4)})"
    elif config['optimizer_type'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            weight_decay=config.get('weight_decay', 1e-4)
        )
        optimizer_info = f"Adam(lr={base_lr}, weight_decay={config.get('weight_decay', 1e-4)})"
    elif config['optimizer_type'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=base_lr,
            weight_decay=config.get('weight_decay', 1e-2)
        )
        optimizer_info = f"AdamW(lr={base_lr}, weight_decay={config.get('weight_decay', 1e-2)})"

    # 学习率调度器
    if config['lr_decay_type'].lower() == 'cos':
        if is_freeze:
            T_max = config['Freeze_Epoch'] - config['Init_Epoch']
        else:
            T_max = config['UnFreeze_Epoch'] - config['Freeze_Epoch']

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config['Min_lr']
        )
        scheduler_info = f"CosineAnnealingLR(T_max={T_max}, eta_min={config['Min_lr']})"
    elif config['lr_decay_type'].lower() == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('lr_gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        scheduler_info = f"StepLR(step_size={step_size}, gamma={gamma})"
    elif config['lr_decay_type'].lower() == 'plateau':
        patience = config.get('patience', 10)
        factor = config.get('factor', 0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            verbose=True
        )
        scheduler_info = f"ReduceLROnPlateau(factor={factor}, patience={patience})"

    return optimizer, scheduler, current_batch_size, optimizer_info, scheduler_info