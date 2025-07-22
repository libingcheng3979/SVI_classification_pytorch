"""
默认配置参数
"""

def get_default_config():
    """返回默认配置参数"""
    config = {
        "num_classes": 4,
        "backbone": "resnet50",
        "model_path": r"model_data/resnet50_places365.pth.tar",
        "dataset_path": r"datasets",
        "input_shape": [192, 512],
        "Init_Epoch": 0,
        "Freeze_Epoch": 50,
        "UnFreeze_Epoch": 200,
        "Freeze_batch_size": 32,
        "Unfreeze_batch_size": 32,
        "Freeze_Train": True,
        "Init_lr": 0.01,
        "Min_lr": 0.0001,
        "optimizer_type": "sgd",
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "lr_decay_type": "cos",
        "save_period": 10,
        "save_dir": "logs",
        "num_workers": 4,
        "vis_frequency": 10,  # 每隔多少个epoch可视化一次
        "grad_clip": 1.0,  # 梯度裁剪阈值
        "early_stopping_patience": 10,  # 早停耐心值
        # GradCAM相关参数
        "gradcam_samples": 10,  # GradCAM可视化的样本数量
        "generate_gradcam": True  # 是否生成GradCAM可视化
    }
    return config

def print_config(config):
    """打印配置信息"""
    print("\n" + "=" * 70)
    print("Street Scene Classification - Training Configuration")
    print("=" * 70)
    print(f"{'Parameter':<25} | {'Value':<40}")
    print("-" * 70)
    for key, value in config.items():
        print(f"{key:<25} | {str(value):<40}")
    print("=" * 70 + "\n")