"""
独立的GradCAM生成脚本
"""

import os
import torch
from torchvision import transforms
from PIL import Image

from config.default_config import get_default_config
from models.resnet import build_model
from utils.gradcam import GradCAM, create_save_dir, show_gradcam

def main():
    # 直接设置参数
    model_path = "logs/20250718_145141/checkpoints/best_model.pth"  # 修改为模型路径
    image_path = "gradcam_results/input"  # 修改为图像路径或目录
    output_dir = "gradcam_results"  # 输出目录
    class_names = None  # 如果为None，将尝试从模型加载，或使用默认类别名称

    # 获取配置
    config = get_default_config()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 构建模型
    model = build_model(config, device)

    # 加载模型权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")

        # 如果checkpoint中有类别信息，则使用它
        if 'classes' in checkpoint:
            class_names = checkpoint['classes']
            print(f"Using class names from checkpoint: {class_names}")
        else:
            class_names = class_names if class_names else [f'Class {i}' for i in range(config['num_classes'])]
            print(f"Using provided or default class names: {class_names}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")

    # 创建GradCAM对象
    gradcam = GradCAM(model)

    # 创建保存目录
    save_dir = create_save_dir(output_dir)
    print(f"Visualizations will be saved to: {save_dir}")

    # 数据变换
    data_transform = transforms.Compose([
        transforms.Resize(config['input_shape']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 检查输入是文件还是目录
    if os.path.isfile(image_path):
        image_paths = [image_path]
    elif os.path.isdir(image_path):
        # 获取目录中的所有图像文件
        image_paths = [os.path.join(image_path, f) for f in os.listdir(image_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        raise FileNotFoundError(f"Image path {image_path} not found")

    for i, img_path in enumerate(image_paths):
        # 提取文件名
        img_filename = os.path.basename(img_path).split('.')[0]

        # 加载原图
        original_image = Image.open(img_path).convert('RGB')

        # 准备输入
        img_tensor = data_transform(original_image).unsqueeze(0).to(device)

        # 获取预测
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_idx = torch.argmax(output, dim=1).item()
            prob = probs[0, pred_idx].item()

        # 计算GradCAM
        cam = gradcam(img_tensor, pred_idx)

        # 获取预测类别名称
        pred_class = class_names[pred_idx]
        print(f"Image {i+1}/{len(image_paths)}: Predicted as {pred_class} with probability {prob:.4f}")

        # 显示并保存结果
        show_gradcam(
            original_image,
            cam,
            pred_class,
            prob,
            save_dir,
            f"sample_{i + 1}_{img_filename}",
            display=False  # 不显示图像，只保存
        )

    print(f"All visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main()