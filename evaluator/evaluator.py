"""
评估器类
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

from utils.metrics import compute_confusion_matrix, compute_classification_report, compute_roc_curves, compute_pr_curves
from utils.visualization import (plot_classification_report_table, plot_confusion_matrix,
                                 plot_roc_curves, plot_pr_curves, visualize_misclassifications,
                                 create_experiment_summary)


class Evaluator:
    """
    模型评估器
    """

    def __init__(self, model, val_loader, device, config, logs_dir, figures_dir,
                 checkpoints_dir, class_names, class_to_idx, writer, timestamp,
                 early_stopped, patience):
        """
        初始化评估器

        Args:
            model: 模型
            val_loader: 验证数据加载器
            device: 设备 (cuda/cpu)
            config: 配置字典
            logs_dir: 日志保存目录
            figures_dir: 图像保存目录
            checkpoints_dir: 检查点保存目录
            class_names: 类别名称列表
            class_to_idx: 类别到索引的映射
            writer: TensorBoard SummaryWriter
            timestamp: 时间戳
            early_stopped: 是否早停
            patience: 早停耐心值
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.logs_dir = logs_dir
        self.figures_dir = figures_dir
        self.checkpoints_dir = checkpoints_dir
        self.class_names = class_names
        self.class_to_idx = class_to_idx
        self.writer = writer
        self.timestamp = timestamp
        self.early_stopped = early_stopped
        self.patience = patience

        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        """
        最终模型评估
        """
        print("\n" + "=" * 50)
        print("Performing final evaluation...")
        print("=" * 50)

        # 加载最佳模型
        best_model_path = os.path.join(self.checkpoints_dir, 'best_model.pth')
        try:
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            best_epoch = checkpoint['epoch']
            best_acc = checkpoint['val_acc']

            print(f"Loaded best model from epoch {best_epoch + 1} with validation accuracy {best_acc:.2f}%")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Using current model for evaluation.")
            best_epoch = -1
            best_acc = 0.0

        # 进行评估
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算分类报告
        report, report_df = compute_classification_report(all_labels, all_preds, self.class_names)
        report_path = os.path.join(self.logs_dir, 'classification_report.csv')
        report_df.to_csv(report_path)

        # 记录到TensorBoard
        for class_name in self.class_names:
            if class_name in report:
                self.writer.add_scalar(f'FinalMetrics/Precision_{class_name}', report[class_name]['precision'])
                self.writer.add_scalar(f'FinalMetrics/Recall_{class_name}', report[class_name]['recall'])
                self.writer.add_scalar(f'FinalMetrics/F1_{class_name}', report[class_name]['f1-score'])

        self.writer.add_scalar('FinalMetrics/Accuracy', report['accuracy'])
        self.writer.add_scalar('FinalMetrics/MacroF1', report['macro avg']['f1-score'])
        self.writer.add_scalar('FinalMetrics/WeightedF1', report['weighted avg']['f1-score'])

        # 创建分类报告表格
        plot_classification_report_table(report, self.class_names, self.figures_dir)

        # 计算混淆矩阵
        cm, cm_normalized = compute_confusion_matrix(all_labels, all_preds, self.class_names)

        # 绘制最终混淆矩阵
        plot_confusion_matrix(cm, cm_normalized, self.class_names, best_epoch, self.figures_dir)

        # 重命名最终混淆矩阵文件
        os.rename(os.path.join(self.figures_dir, f'confusion_matrix_epoch_{best_epoch + 1}.png'),
                  os.path.join(self.figures_dir, 'final_confusion_matrix_normalized.png'))

        # 计算并绘制ROC曲线
        fpr, tpr, roc_auc = compute_roc_curves(all_labels, all_probs, self.config['num_classes'])
        plot_roc_curves(fpr, tpr, roc_auc, self.class_names, self.figures_dir, self.writer)

        # 计算并绘制PR曲线
        precision, recall, avg_precision = compute_pr_curves(all_labels, all_probs, self.config['num_classes'])
        plot_pr_curves(precision, recall, avg_precision, self.class_names, self.figures_dir, self.writer)

        # 保存AUC和AP值
        auc_df = pd.DataFrame({
            'Class': self.class_names,
            'AUC': [roc_auc[i] for i in range(self.config['num_classes'])]
        })
        auc_df.to_csv(os.path.join(self.logs_dir, 'auc_values.csv'), index=False)

        ap_df = pd.DataFrame({
            'Class': self.class_names,
            'AP': [avg_precision[i] for i in range(self.config['num_classes'])]
        })
        ap_df.to_csv(os.path.join(self.logs_dir, 'ap_values.csv'), index=False)

        # 可视化错误分类样本
        visualize_misclassifications(self.val_loader.dataset, all_labels, all_preds, self.class_names, self.figures_dir)

        # 保存最终结果
        final_results = {
            'best_epoch': int(best_epoch + 1),
            'best_accuracy': float(best_acc),
            'final_accuracy': float(report['accuracy'] * 100),
            'final_macro_f1': float(report['macro avg']['f1-score']),
            'final_weighted_f1': float(report['weighted avg']['f1-score']),
            'class_reports': {
                class_name: {
                    'precision': float(report[class_name]['precision']),
                    'recall': float(report[class_name]['recall']),
                    'f1-score': float(report[class_name]['f1-score'])
                } for class_name in self.class_names
            },
            'training_time': str(self.timestamp),
            'config': self.config,
            'early_stopped': self.early_stopped
        }

        with open(os.path.join(self.logs_dir, 'final_results.json'), 'w') as f:
            import json
            json.dump(final_results, f, indent=4)

        # 创建实验结果一页纸摘要
        create_experiment_summary(self.config, self.logs_dir, self.figures_dir,
                                  self.class_names, self.timestamp, self.early_stopped, self.patience)

        print(f"Final evaluation completed. Results saved to {self.logs_dir}")

        return final_results