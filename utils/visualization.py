"""
可视化工具函数
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import matplotlib.font_manager as fm

# 设置中文字体，需要替换为您系统中已安装的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以换成'Microsoft YaHei'等
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def create_config_table(config, figures_dir):
    """
    创建配置信息表格图像

    Args:
        config: 配置字典
        figures_dir: 图像保存目录
    """
    # 创建表格数据
    data = []
    for key, value in config.items():
        data.append([key, str(value)])

    # 创建表格
    plt.figure(figsize=(12, len(data) * 0.4))
    table = plt.table(cellText=data,
                      colLabels=["Parameter", "Value"],
                      loc='center',
                      cellLoc='left',
                      colWidths=[0.3, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.axis('off')
    plt.title("Experiment Configuration", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'config_table.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, cm_normalized, class_names, epoch, figures_dir, writer=None):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        cm_normalized: 归一化混淆矩阵
        class_names: 类别名称列表
        epoch: 当前轮次
        figures_dir: 图像保存目录
        writer: TensorBoard SummaryWriter
    """
    # 绘制标准混淆矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=f'Confusion Matrix - Epoch {epoch + 1}',
           ylabel='True label',
           xlabel='Predicted label')

    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # 添加到TensorBoard
    if writer is not None:
        writer.add_figure('ConfusionMatrix/Raw', fig, epoch)
    plt.close(fig)

    # 绘制归一化混淆矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm_normalized.shape[1]),
           yticks=np.arange(cm_normalized.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=f'Normalized Confusion Matrix - Epoch {epoch + 1}',
           ylabel='True label',
           xlabel='Predicted label')

    # 添加百分比标签
    thresh = 0.5
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, format(cm_normalized[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    fig.tight_layout()

    # 添加到TensorBoard
    if writer is not None:
        writer.add_figure('ConfusionMatrix/Normalized', fig, epoch)
    plt.close(fig)

    # 保存到文件
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Normalized Confusion Matrix - Epoch {epoch + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'confusion_matrix_epoch_{epoch + 1}.png'), dpi=300)
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, figures_dir, logs_dir):
    """
    绘制训练曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表
        figures_dir: 图像保存目录
        logs_dir: 日志保存目录
    """
    epochs = list(range(1, len(train_losses) + 1))

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'training_curves.png'), dpi=300)
    plt.close()

    # 保存训练历史到CSV
    history_df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    })
    history_df.to_csv(os.path.join(logs_dir, 'training_history.csv'), index=False)


def plot_classification_report_table(report, class_names, figures_dir):
    """
    将分类报告绘制为表格图像

    Args:
        report: 分类报告字典
        class_names: 类别名称列表
        figures_dir: 图像保存目录
    """
    # 提取表格数据
    rows = []
    for class_name in class_names + ['macro avg', 'weighted avg']:
        if class_name in report:
            rows.append([
                class_name,
                f"{report[class_name]['precision']:.4f}",
                f"{report[class_name]['recall']:.4f}",
                f"{report[class_name]['f1-score']:.4f}",
                f"{report[class_name]['support']}"
            ])

    # 添加精度行
    rows.append(['accuracy', f"{report['accuracy']:.4f}", '', '',
                 f"{sum([report[c]['support'] for c in class_names])}"])

    # 创建表格
    plt.figure(figsize=(10, len(rows) * 0.5))
    table = plt.table(cellText=rows,
                      colLabels=['', 'Precision', 'Recall', 'F1-Score', 'Support'],
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.axis('off')
    plt.title("Classification Report", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'classification_report.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(fpr, tpr, roc_auc, class_names, figures_dir, writer=None):
    """
    绘制ROC曲线

    Args:
        fpr: 假阳性率
        tpr: 真阳性率
        roc_auc: ROC曲线下面积
        class_names: 类别名称列表
        figures_dir: 图像保存目录
        writer: TensorBoard SummaryWriter
    """
    # 绘制每个类的ROC曲线
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, color, class_name in zip(range(len(class_names)), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_name} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'roc_curves.png'), dpi=300)
    plt.close()

    # 添加到TensorBoard
    if writer is not None:
        roc_fig = plt.figure(figsize=(10, 8))
        for i, color, class_name in zip(range(len(class_names)), colors, class_names):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{class_name} (AUC = {roc_auc[i]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        writer.add_figure('FinalMetrics/ROC', roc_fig)
        plt.close(roc_fig)


def plot_pr_curves(precision, recall, avg_precision, class_names, figures_dir, writer=None):
    """
    绘制精确率-召回率曲线

    Args:
        precision: 精确率
        recall: 召回率
        avg_precision: 平均精确率
        class_names: 类别名称列表
        figures_dir: 图像保存目录
        writer: TensorBoard SummaryWriter
    """
    # 绘制每个类的PR曲线
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, color, class_name in zip(range(len(class_names)), colors, class_names):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{class_name} (AP = {avg_precision[i]:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pr_curves.png'), dpi=300)
    plt.close()

    # 添加到TensorBoard
    if writer is not None:
        pr_fig = plt.figure(figsize=(10, 8))
        for i, color, class_name in zip(range(len(class_names)), colors, class_names):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                     label=f'{class_name} (AP = {avg_precision[i]:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curves')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.7)
        writer.add_figure('FinalMetrics/PR', pr_fig)
        plt.close(pr_fig)


def visualize_misclassifications(val_dataset, y_true, y_pred, class_names, figures_dir):
    """
    可视化错误分类的样本

    Args:
        val_dataset: 验证数据集
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        figures_dir: 图像保存目录
    """
    # 获取验证集的所有图像路径和标签
    all_imgs = [val_dataset.imgs[i][0] for i in range(len(val_dataset))]

    # 找出错误分类的样本
    misclassified_indices = np.where(np.array(y_true) != np.array(y_pred))[0]

    if len(misclassified_indices) == 0:
        print("No misclassified samples found.")
        return

    # 限制可视化的错误样本数量
    max_samples = min(20, len(misclassified_indices))
    indices_to_plot = np.random.choice(misclassified_indices, max_samples, replace=False)

    # 准备绘图
    num_cols = 5
    num_rows = (max_samples + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    axes = axes.flatten()

    valid_samples = 0
    for i, idx in enumerate(indices_to_plot):
        if i < len(axes):
            try:
                img_path = all_imgs[idx]
                # 检查文件是否存在
                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found: {img_path}")
                    continue

                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Failed to read image: {img_path}")
                    continue

                # 转换颜色空间
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                true_label = class_names[y_true[idx]]
                pred_label = class_names[y_pred[idx]]

                axes[i].imshow(img)
                axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
                axes[i].axis('off')
                valid_samples += 1

            except Exception as e:
                print(f"Error processing image at index {idx}: {str(e)}")
                continue

    # 如果没有有效样本，则提前返回
    if valid_samples == 0:
        plt.close(fig)
        print("No valid samples to visualize.")
        return

    # 隐藏未使用的子图
    for i in range(valid_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'misclassified_samples.png')
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved misclassified samples visualization to: {save_path}")
    except Exception as e:
        print(f"Error saving figure: {str(e)}")
    finally:
        plt.close(fig)


def create_experiment_summary(config, logs_dir, figures_dir, class_names, timestamp, early_stopped, patience):
    """
    创建实验结果一页纸摘要

    Args:
        config: 配置字典
        logs_dir: 日志保存目录
        figures_dir: 图像保存目录
        class_names: 类别名称列表
        timestamp: 时间戳
        early_stopped: 是否早停
        patience: 早停耐心值
    """
    # 加载训练历史
    try:
        history_df = pd.read_csv(os.path.join(logs_dir, 'training_history.csv'))
    except Exception as e:
        print(f"Error loading training history: {e}")
        return

    # 加载分类报告
    try:
        report_df = pd.read_csv(os.path.join(logs_dir, 'classification_report.csv'))
    except Exception as e:
        print(f"Error loading classification report: {e}")
        return

    # 创建摘要图
    plt.figure(figsize=(12, 16))

    # 1. 标题
    plt.subplot(5, 1, 1)
    plt.axis('off')
    plt.text(0.5, 0.8, 'Street Scene Classification', fontsize=20, ha='center', weight='bold')
    plt.text(0.5, 0.5, f'Experiment Summary - {timestamp}', fontsize=16, ha='center')
    plt.text(0.5, 0.2, f'Model: {config["backbone"]} (Places365 transfer learning)', fontsize=14, ha='center')

    # 2. 训练曲线
    plt.subplot(5, 2, 3)
    plt.plot(history_df['epoch'], history_df['train_loss'], 'b-', label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(5, 2, 4)
    plt.plot(history_df['epoch'], history_df['train_accuracy'], 'b-', label='Train Acc')
    plt.plot(history_df['epoch'], history_df['val_accuracy'], 'r-', label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    # 3. 混淆矩阵
    try:
        plt.subplot(5, 2, 5)
        cm_img = plt.imread(os.path.join(figures_dir, 'final_confusion_matrix_normalized.png'))
        plt.imshow(cm_img)
        plt.title('Confusion Matrix')
        plt.axis('off')
    except Exception as e:
        print(f"Error loading confusion matrix image: {e}")

    # 4. ROC曲线
    try:
        plt.subplot(5, 2, 6)
        roc_img = plt.imread(os.path.join(figures_dir, 'roc_curves.png'))
        plt.imshow(roc_img)
        plt.title('ROC Curves')
        plt.axis('off')
    except Exception as e:
        print(f"Error loading ROC curves image: {e}")

    # 5. 关键指标
    plt.subplot(5, 1, 4)
    plt.axis('off')

    # 提取关键指标
    try:
        # 从正确的列中提取accuracy
        accuracy = report_df.loc[report_df['Unnamed: 0'] == 'accuracy', 'precision'].values[0] * 100

        # 提取宏观F1和加权F1
        macro_f1 = report_df.loc[report_df['Unnamed: 0'] == 'macro avg', 'f1-score'].values[0]
        weighted_f1 = report_df.loc[report_df['Unnamed: 0'] == 'weighted avg', 'f1-score'].values[0]

        plt.text(0.5, 0.9, 'Key Metrics', fontsize=16, ha='center', weight='bold')
        plt.text(0.5, 0.75, f'Accuracy: {accuracy:.2f}%', fontsize=14, ha='center')
        plt.text(0.5, 0.6, f'Macro F1: {macro_f1:.4f}', fontsize=14, ha='center')
        plt.text(0.5, 0.45, f'Weighted F1: {weighted_f1:.4f}', fontsize=14, ha='center')

        # 添加类别指标
        y_pos = 0.3
        for class_name in class_names:
            class_data = report_df.loc[report_df['Unnamed: 0'] == class_name]
            if not class_data.empty:
                precision = class_data['precision'].values[0]
                recall = class_data['recall'].values[0]
                f1 = class_data['f1-score'].values[0]
                plt.text(0.5, y_pos,
                         f"Class '{class_name}': Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}",
                         fontsize=12, ha='center')
                y_pos -= 0.15
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        plt.text(0.5, 0.5, "Error extracting metrics", fontsize=14, ha='center', color='red')

    # 6. 配置信息
    plt.subplot(5, 1, 5)
    plt.axis('off')
    plt.text(0.5, 0.9, 'Configuration', fontsize=16, ha='center', weight='bold')

    info_text = (
        f"Backbone: {config['backbone']}\n"
        f"Input Shape: {config['input_shape']}\n"
        f"Batch Size: {config['Freeze_batch_size']} / {config['Unfreeze_batch_size']}\n"
        f"Learning Rate: {config['Init_lr']} → {config['Min_lr']}\n"
        f"Optimizer: {config['optimizer_type']}\n"
        f"LR Scheduler: {config['lr_decay_type']}\n"
        f"Training: {config['Freeze_Epoch']} epochs frozen + {config['UnFreeze_Epoch'] - config['Freeze_Epoch']} epochs unfrozen\n"
        f"Dataset: {config['num_train']} train, {config['num_val']} validation\n"
        f"Early Stopping: {'启用' if early_stopped else '未触发'} (patience={patience})"
    )

    plt.text(0.5, 0.6, info_text, fontsize=12, ha='center', va='center', linespacing=1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, 'experiment_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created experiment summary at {os.path.join(logs_dir, 'experiment_summary.png')}")