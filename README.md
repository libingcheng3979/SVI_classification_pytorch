# 街景分类与 GradCAM 可视化

这是一个基于深度学习的街景分类项目，利用 GradCAM 技术对模型预测进行可视化分析。项目支持从数据加载、训练、验证到模型评估的全流程，并提供了丰富的可视化功能（如混淆矩阵、ROC 曲线、PR 曲线、分类报告等）。此外，项目支持早停机制、多阶段训练（冻结和解冻）、学习率调度及 GradCAM 热图生成。

## 项目特点

- 🚀 支持基于ResNet的预训练模型架构  
- 📊 完整的训练和评估流程
- 🔍 集成GradCAM可视化，帮助理解模型决策
- 📈 详细的训练日志和TensorBoard支持

## 项目结构

```
SVI_classification_pytorch
├── config
│   └── default_config.py       # 配置参数
├── data
│   ├── dataset.py              # 数据集加载工具
│   └── transforms.py           # 数据增强转换
├── datasets                    # 数据集目录
│   ├── test                    # 测试数据集，包含类别文件夹
│   └── train                   # 训练数据集，包含类别文件夹
├── evaluator
│   └── evaluator.py            # 模型评估逻辑
├── generate_gradcam.py         # 独立的GradCAM生成脚本
├── gradcam_results             # GradCAM可视化输出
│   └── input                   # GradCAM输入图像
├── logs                        # 训练日志和检查点
├── main.py                     # 主训练脚本
├── model_data
│   └── resnet50_places365.pth.tar  # 预训练权重
├── models
│   └── resnet.py               # 模型定义和设置
├── trainer
│   └── trainer.py              # 训练循环实现
└── utils                       # 工具函数
    ├── early_stopping.py       # 早停实现
    ├── gradcam.py              # GradCAM可视化工具
    ├── metrics.py              # 评估指标计算
    └── visualization.py        # 结果可视化工具
```

## 使用方法

### 数据准备

将数据集组织为以下结构：

```
datasets/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── class2/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── ...
```
### 训练模型

使用默认参数训练模型：

```bash
python main.py
```

训练过程包括：
1. 加载和预处理数据集
2. 使用Places365预训练权重初始化模型
3. 冻结训练阶段（仅训练FC层）
4. 解冻训练阶段（全模型微调）
5. 评估和生成可视化结果

### GradCAM可视化

为特定图像生成GradCAM可视化：

```bash
python generate_gradcam.py
```

你可以在脚本中修改以下参数：
- `model_path`：训练好的模型检查点路径
- `image_path`：输入图像或目录的路径
- `output_dir`：保存可视化结果的目录

GradCAM可视化帮助解释模型的决策过程，展示模型关注的图像区域：

![sample_5_0_38_79580.png](gradcam_results%2F20250721_223755%2Fsample_5_0_38_79580.png)

左图：原始图像，中图：热力图，右图：叠加后的图像

## 自定义配置

所有配置参数存放在 `config/default_config.py` 文件中，可根据需求进行修改。以下为部分关键参数：

| 参数名称            | 默认值        | 描述                              |
|---------------------|---------------|-----------------------------------|
| `num_classes`       | `4`           | 类别数量                          |
| `input_shape`       | `[192, 512]`  | 输入图像尺寸                      |
| `Init_lr`           | `0.01`        | 初始学习率                       |
| `Freeze_Epoch`      | `50`          | 冻结阶段训练的 epoch 数量         |
| `UnFreeze_Epoch`    | `200`         | 解冻阶段训练的 epoch 数量         |
| `optimizer_type`    | `sgd`         | 优化器类型（支持 sgd、adam、adamw）|
| `gradcam_samples`   | `10`          | GradCAM 可视化样本数量           |

## 其他功能

1. **早停机制**  
   基于验证损失的早停策略，避免过拟合。

2. **灵活的训练配置**  
   支持冻结与解冻两阶段训练，优化器与学习率调度器均可灵活配置。

3. **丰富的评估指标**  
   自动生成分类报告、混淆矩阵、ROC 曲线、PR 曲线。

4. **实验总结**  
   生成包含所有关键结果的实验摘要图表，方便快速分析。

## 参考资料
- https://github.com/CSAILVision/places365
- https://github.com/jacobgil/pytorch-grad-cam

## 声明
有任何疑问欢迎联系交流（libingcheng3979@163.com).