"""
GradCAM可视化工具
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import os
from datetime import datetime

# 设置matplotlib使用Arial字体
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12


class GradCAM:
    def __init__(self, model):
        """
        初始化GradCAM

        Args:
            model: 用于生成GradCAM的模型
        """
        self.model = model
        self.model.eval()

        # 获取最后一个卷积层
        self.target_layer = model.layer4[-1].conv3

        # 存储梯度和激活值
        self.gradients = None
        self.activations = None

        # 注册钩子
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations = output
            return None

        # 使用register_full_backward_hook
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, img_tensor, class_idx=None):
        """
        生成GradCAM热图

        Args:
            img_tensor: 输入图像张量
            class_idx: 目标类别索引，如果为None则使用预测的类别

        Returns:
            cam: GradCAM热图
        """
        # 前向传播
        output = self.model(img_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output).item()

        # 反向传播
        self.model.zero_grad()
        output[0, class_idx].backward()

        # 计算GradCAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # 全局平均池化梯度
        weights = np.mean(gradients, axis=(1, 2))

        # 加权激活
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = np.maximum(cam, 0)

        # 归一化
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam


def create_save_dir(base_dir="gradcam_results"):
    """
    创建保存GradCAM结果的目录

    Args:
        base_dir: 基础目录

    Returns:
        save_dir: 创建的保存目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def show_gradcam(original_image, cam, class_name, prob, save_dir=None, img_name=None, true_class=None, display=False):
    """
    显示GradCAM结果并保存

    Args:
        original_image: 原始图像
        cam: GradCAM热图
        class_name: 预测的类别名称
        prob: 预测的概率
        save_dir: 保存目录
        img_name: 图像名称
        true_class: 真实类别名称
        display: 是否显示图像，默认为False
    """
    # 调整热图大小
    cam = cv2.resize(cam, (original_image.width, original_image.height))

    # 转换为热图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 转换原始图像为numpy数组
    np_img = np.array(original_image)

    # 叠加
    superimposed = np.uint8(0.6 * np_img + 0.4 * heatmap)

    # 创建图像和子图
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # 设置图像标题 - 使用更便于修改的变量
    original_title = ''
    heatmap_title = ''
    overlay_title = ''

    # 原始图像
    ax[0].imshow(np_img)
    ax[0].set_title(original_title, fontname='Arial')
    ax[0].axis('off')

    # 热图
    ax[1].imshow(heatmap)
    ax[1].set_title(heatmap_title, fontname='Arial')
    ax[1].axis('off')

    # 叠加图像
    ax[2].imshow(superimposed)
    ax[2].set_title(overlay_title, fontname='Arial')
    ax[2].axis('off')

    plt.tight_layout()

    # 如果提供了保存目录和图像名称，则保存图像
    if save_dir and img_name:
        save_path = os.path.join(save_dir, f"{img_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to: {save_path}")

    # 只有当display=True时才显示图像
    if display:
        plt.show()
    else:
        plt.close(fig)


def generate_gradcam_visualizations(model, val_dataset, class_names, device, num_samples=10, save_dir=None):
    """
    为验证集中的随机样本生成GradCAM可视化

    Args:
        model: 模型
        val_dataset: 验证集
        class_names: 类别名称列表
        device: 设备
        num_samples: 要可视化的样本数量
        save_dir: 保存目录
    """
    # 创建GradCAM对象
    gradcam = GradCAM(model)

    # 如果没有提供保存目录，则创建一个
    if save_dir is None:
        save_dir = create_save_dir()
    print(f"Visualizations will be saved to: {save_dir}")

    # 随机选择图片
    random_indices = np.random.choice(len(val_dataset), num_samples, replace=False)

    for i, idx in enumerate(random_indices):
        # 获取图像和标签
        img, label = val_dataset[idx]
        img_path = val_dataset.imgs[idx][0]

        # 提取文件名作为保存名称的一部分
        img_filename = os.path.basename(img_path).split('.')[0]

        # 加载原图
        original_image = Image.open(img_path).convert('RGB')

        # 准备输入
        img_tensor = img.unsqueeze(0).to(device)

        # 获取预测
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_idx = torch.argmax(output, dim=1).item()
            prob = probs[0, pred_idx].item()

        # 计算GradCAM
        cam = gradcam(img_tensor, pred_idx)

        # 获取类别名称
        true_class = class_names[label]
        pred_class = class_names[pred_idx]
        title = f"True: {true_class}, Pred: {pred_class}"
        print(title)

        # 显示并保存结果
        show_gradcam(
            original_image,
            cam,
            pred_class,
            prob,
            save_dir,
            f"sample_{i + 1}_{img_filename}",
            true_class
        )

    print(f"All visualizations have been saved to: {save_dir}")