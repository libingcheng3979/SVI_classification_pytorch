"""
训练器类
"""

import os
import time
import datetime
import torch
import torch.nn as nn
import json
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.early_stopping import EarlyStopping
from utils.metrics import compute_confusion_matrix
from utils.visualization import plot_confusion_matrix, plot_training_curves


class Trainer:
    """
    模型训练器
    """

    def __init__(self, model, train_loader, val_loader, device, config, logs_dir, checkpoints_dir,
                 tb_dir, figures_dir, class_names, class_to_idx, timestamp):
        """
        初始化训练器

        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备 (cuda/cpu)
            config: 配置字典
            logs_dir: 日志保存目录
            checkpoints_dir: 检查点保存目录
            tb_dir: TensorBoard日志目录
            figures_dir: 图像保存目录
            class_names: 类别名称列表
            class_to_idx: 类别到索引的映射
            timestamp: 时间戳
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.logs_dir = logs_dir
        self.checkpoints_dir = checkpoints_dir
        self.tb_dir = tb_dir
        self.figures_dir = figures_dir
        self.class_names = class_names
        self.class_to_idx = class_to_idx
        self.timestamp = timestamp

        # 配置TensorBoard
        self.writer = SummaryWriter(log_dir=tb_dir)

        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_acc = 0.0
        self.current_epoch = self.config['Init_Epoch']

        # 早停设置
        self.early_stopping = EarlyStopping(patience=self.config.get('early_stopping_patience', 15), verbose=True)
        self.early_stopped = False

    def train_one_epoch(self, epoch, optimizer):
        """
        训练一个epoch

        Args:
            epoch: 当前epoch
            optimizer: 优化器

        Returns:
            epoch_loss: 当前epoch的平均损失
            epoch_acc: 当前epoch的平均准确率
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.current_end_epoch}")

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('grad_clip', 1.0))

            # 更新参数
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total,
                'lr': optimizer.param_groups[0]['lr']
            })

            # 记录到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Training/BatchLoss', loss.item(), global_step)
            self.writer.add_scalar('Training/BatchAcc', 100. * correct / total, global_step)
            self.writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], global_step)

            # 定期记录参数直方图
            if batch_idx % 100 == 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.writer.add_histogram(f'Parameters/{name}', param.data, global_step)
                        self.writer.add_histogram(f'Gradients/{name}', param.grad.data, global_step)

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)

        # 记录到TensorBoard
        self.writer.add_scalar('Epoch/TrainLoss', epoch_loss, epoch)
        self.writer.add_scalar('Epoch/TrainAcc', epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """
        验证模型

        Args:
            epoch: 当前epoch

        Returns:
            epoch_loss: 当前epoch的平均损失
            epoch_acc: 当前epoch的平均准确率
            all_preds: 所有预测标签
            all_labels: 所有真实标签
            all_probs: 所有预测概率
            early_stop: 是否应该早停
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # 计算概率
                probs = torch.nn.functional.softmax(outputs, dim=1)

                # 统计
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)

        # 记录到TensorBoard
        self.writer.add_scalar('Epoch/ValLoss', epoch_loss, epoch)
        self.writer.add_scalar('Epoch/ValAcc', epoch_acc, epoch)

        # 早停检查
        early_stop = self.early_stopping(epoch_loss, self.writer, epoch)
        if early_stop:
            self.early_stopped = True

        # 每N个epoch可视化混淆矩阵
        if (epoch + 1) % self.config.get('vis_frequency', 10) == 0 or epoch == self.current_end_epoch - 1:
            cm, cm_normalized = compute_confusion_matrix(all_labels, all_preds, self.class_names)

            # 绘制混淆矩阵
            plot_confusion_matrix(cm, cm_normalized, self.class_names, epoch, self.figures_dir, self.writer)

        # 保存最佳模型
        if epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': epoch_acc,
                'val_loss': epoch_loss,
                'classes': self.class_names,
                'class_to_idx': self.class_to_idx
            }, os.path.join(self.checkpoints_dir, 'best_model.pth'))
            print(f"Best model saved with accuracy: {self.best_acc:.2f}%")

        # 定期保存模型
        if (epoch + 1) % self.config['save_period'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': epoch_acc,
                'val_loss': epoch_loss,
                'classes': self.class_names,
                'class_to_idx': self.class_to_idx
            }, os.path.join(self.checkpoints_dir, f'ep{epoch + 1:03d}_acc{epoch_acc:.2f}.pth'))

        return epoch_loss, epoch_acc, all_preds, all_labels, all_probs, early_stop

    def train(self, optimizer, scheduler, current_batch_size, optimizer_info, scheduler_info):
        """
        训练模型

        Args:
            optimizer: 优化器
            scheduler: 学习率调度器
            current_batch_size: 当前批次大小
            optimizer_info: 优化器信息
            scheduler_info: 学习率调度器信息

        Returns:
            is_early_stopped: 是否触发早停
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.current_batch_size = current_batch_size

        # 记录优化器和学习率调度器信息到TensorBoard
        self.writer.add_text('Training/Optimizer', optimizer_info)
        self.writer.add_text('Training/Scheduler', scheduler_info)
        self.writer.add_text('Training/ParamGroups', str(len(optimizer.param_groups)))

        for epoch in range(self.current_epoch, self.current_end_epoch):
            self.current_epoch = epoch

            # 训练一个epoch
            train_loss, train_acc = self.train_one_epoch(epoch, optimizer)

            # 验证
            val_loss, val_acc, all_preds, all_labels, all_probs, early_stop = self.validate(epoch)

            # 检查是否触发早停
            if early_stop:
                print(f"阶段训练提前终止")
                return True

            # 更新学习率
            if self.config['lr_decay_type'].lower() == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # 打印结果
            print(f"Epoch {epoch + 1}/{self.current_end_epoch} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # 每epoch结束绘制训练曲线
            plot_training_curves(self.train_losses, self.val_losses,
                                 self.train_accuracies, self.val_accuracies,
                                 self.figures_dir, self.logs_dir)

        return False

    def set_current_end_epoch(self, end_epoch):
        """设置当前结束的epoch"""
        self.current_end_epoch = end_epoch

    def reset_early_stopping(self):
        """重置早停计数器"""
        self.early_stopping = EarlyStopping(patience=self.config.get('early_stopping_patience', 15), verbose=True)