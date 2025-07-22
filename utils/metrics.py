"""
评估指标和计算函数
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score
from sklearn.preprocessing import label_binarize


def compute_confusion_matrix(y_true, y_pred, class_names):
    """
    计算混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    Returns:
        cm: 混淆矩阵
        cm_normalized: 归一化混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm, cm_normalized


def compute_classification_report(y_true, y_pred, class_names):
    """
    计算分类报告

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    Returns:
        report: 分类报告字典
        report_df: 分类报告数据框
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return report, report_df


def compute_roc_curves(y_true, y_probs, num_classes):
    """
    计算ROC曲线

    Args:
        y_true: 真实标签
        y_probs: 预测概率
        num_classes: 类别数量

    Returns:
        fpr: 假阳性率
        tpr: 真阳性率
        roc_auc: ROC曲线下面积
    """
    # 将标签进行one-hot编码
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # 计算每个类的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(y_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def compute_pr_curves(y_true, y_probs, num_classes):
    """
    计算PR曲线

    Args:
        y_true: 真实标签
        y_probs: 预测概率
        num_classes: 类别数量

    Returns:
        precision: 精确率
        recall: 召回率
        avg_precision: 平均精确率
    """
    # 将标签进行one-hot编码
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # 计算每个类的PR曲线和AP
    precision = dict()
    recall = dict()
    avg_precision = dict()

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], np.array(y_probs)[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], np.array(y_probs)[:, i])

    return precision, recall, avg_precision