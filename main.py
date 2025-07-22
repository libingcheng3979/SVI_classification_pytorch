"""
街景分类器主程序
"""

import os
import time
import datetime
import torch
import json
import shutil
from torch.utils.tensorboard import SummaryWriter

# 导入项目模块
from config.default_config import get_default_config, print_config
from data.transforms import get_data_transforms
from data.dataset import load_datasets, visualize_dataset_samples
from models.resnet import build_model, setup_optimizer
from trainer.trainer import Trainer
from evaluator.evaluator import Evaluator
from utils.visualization import create_config_table
from utils.gradcam import generate_gradcam_visualizations  # 新增导入

def main():
    """主函数"""
    # 获取配置
    config = get_default_config()

    # 打印配置信息
    print_config(config)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建保存目录
    logs_dir = os.path.join(config['save_dir'], timestamp)
    checkpoints_dir = os.path.join(logs_dir, 'checkpoints')
    tb_dir = os.path.join(logs_dir, 'tensorboard')
    figures_dir = os.path.join(logs_dir, 'figures')
    gradcam_dir = os.path.join(logs_dir, 'gradcam')  # 新增GradCAM保存目录

    for dir_path in [logs_dir, checkpoints_dir, tb_dir, figures_dir, gradcam_dir]:  # 添加gradcam_dir
        os.makedirs(dir_path, exist_ok=True)

    # 配置TensorBoard
    writer = SummaryWriter(log_dir=tb_dir)

    # 保存配置信息
    config_path = os.path.join(logs_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # 创建配置表格
    create_config_table(config, figures_dir)

    # 获取数据转换
    data_transforms = get_data_transforms(config['input_shape'])

    # 加载数据集
    train_loader, val_loader, train_dataset, val_dataset, class_names, class_to_idx = load_datasets(config, data_transforms)

    # 可视化数据集样本
    visualize_dataset_samples(train_dataset, 'train', class_names, writer)
    visualize_dataset_samples(val_dataset, 'val', class_names, writer)

    # 构建模型
    model = build_model(config, device)

    # 记录训练开始时间
    start_time = time.time()
    writer.add_text('Training/StartTime', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 添加模型结构到TensorBoard
    dummy_input = torch.randn(1, 3, config['input_shape'][0], config['input_shape'][1]).to(device)
    writer.add_graph(model, dummy_input)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        tb_dir=tb_dir,
        figures_dir=figures_dir,
        class_names=class_names,
        class_to_idx=class_to_idx,
        timestamp=timestamp
    )

    early_stopped = False

    # 冻结训练阶段
    if config['Freeze_Train'] and config['Freeze_Epoch'] > config['Init_Epoch']:
        print("\n" + "=" * 50)
        print("Freeze training stage...")
        print("=" * 50)

        # 设置优化器
        optimizer, scheduler, current_batch_size, optimizer_info, scheduler_info = setup_optimizer(model, config, is_freeze=True)
        writer.add_text('Training/Stage1', f"Freeze Training: {optimizer_info}, {scheduler_info}")

        # 设置当前结束的epoch
        trainer.set_current_end_epoch(config['Freeze_Epoch'])

        # 开始训练
        early_stopped = trainer.train(optimizer, scheduler, current_batch_size, optimizer_info, scheduler_info)

    # 如果在冻结阶段没有触发早停，则进行解冻训练
    if not early_stopped:
        # 解冻训练阶段
        if config['UnFreeze_Epoch'] > config['Freeze_Epoch']:
            print("\n" + "=" * 50)
            print("Unfreeze training stage...")
            print("=" * 50)

            # 重置早停
            trainer.reset_early_stopping()

            # 设置优化器
            optimizer, scheduler, current_batch_size, optimizer_info, scheduler_info = setup_optimizer(model, config, is_freeze=False)
            writer.add_text('Training/Stage2', f"Unfreeze Training: {optimizer_info}, {scheduler_info}")

            # 设置当前结束的epoch
            trainer.set_current_end_epoch(config['UnFreeze_Epoch'])

            # 开始训练
            early_stopped = trainer.train(optimizer, scheduler, current_batch_size, optimizer_info, scheduler_info)

    # 训练结束，计算训练时间
    end_time = time.time()
    training_time = end_time - start_time
    training_time_str = str(datetime.timedelta(seconds=int(training_time)))

    print(f"\nTraining completed in {training_time_str}")
    print(f"Best validation accuracy: {trainer.best_acc:.2f}%")

    # 记录训练结束信息
    writer.add_text('Training/EndTime', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    writer.add_text('Training/Duration', training_time_str)
    writer.add_text('Training/BestAcc', f"{trainer.best_acc:.2f}%")
    writer.add_text('Training/EarlyStopped', str(early_stopped))

    # 创建评估器并进行最终评估
    evaluator = Evaluator(
        model=model,
        val_loader=val_loader,
        device=device,
        config=config,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        checkpoints_dir=checkpoints_dir,
        class_names=class_names,
        class_to_idx=class_to_idx,
        writer=writer,
        timestamp=timestamp,
        early_stopped=early_stopped,
        patience=config.get('early_stopping_patience', 15)
    )

    # 进行最终评估
    final_results = evaluator.evaluate()

    # 生成GradCAM可视化（新增部分）
    print("\n" + "=" * 50)
    print("Generating GradCAM visualizations...")
    print("=" * 50)

    # 加载最佳模型
    best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1} for GradCAM visualization")

    # 生成GradCAM可视化
    generate_gradcam_visualizations(
        model=model,
        val_dataset=val_dataset,
        class_names=class_names,
        device=device,
        num_samples=10,  # 可以通过配置文件调整
        save_dir=gradcam_dir
    )

    # 关闭TensorBoard写入器
    writer.close()

    print("Street Scene Classification training, evaluation and visualization completed!")

if __name__ == "__main__":
    main()