"""
完整训练脚本（仅训练 + 验证循环）

超参数和实现与 train_full.py 完全对齐，已验证可以收敛。

运行指令：
cd 到 PhysECD 项目根目录下，执行：
python scripts/03_train.py
"""

import os
import sys
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 将项目根目录加入 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physecd.models import PhysECDModel
from physecd.physics import PhysECDLoss


def setup_logging(log_dir):
    """初始化日志配置，同时输出到文件和控制台。"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_data(data_dir, batch_size=64, num_workers=16):
    """加载训练集和验证集。"""
    logger = logging.getLogger(__name__)

    train_path = os.path.join(data_dir, 'train.pt')
    val_path = os.path.join(data_dir, 'val.pt')

    logger.info(f"Loading training data from {train_path}")
    train_data = torch.load(train_path, weights_only=False)
    logger.info(f"Loaded {len(train_data)} training samples")

    logger.info(f"Loading validation data from {val_path}")
    val_data = torch.load(val_path, weights_only=False)
    logger.info(f"Loaded {len(val_data)} validation samples")

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def compute_raw_losses(pred, target, n_states=20):
    """
    计算原始损失（未归一化的 MSE），用于监控各物理量的绝对误差。

    Returns:
        dict: 包含原始损失的字典
    """
    import torch.nn.functional as F

    batch_size = pred['E_pred'].shape[0]

    # 1. 激发能原始 MSE
    y_E = target['y_E'].reshape(batch_size, n_states)
    loss_E_raw = F.mse_loss(pred['E_pred'], y_E)

    # 2. 速度电跃迁偶极原始 MSE（相位不变）
    y_mu_vel = target['y_mu_vel'].reshape(batch_size, n_states, 3)
    mu_vel_diff = pred['mu_total_vel'] - y_mu_vel
    mu_vel_sum = pred['mu_total_vel'] + y_mu_vel
    loss_mu_vel_phase1 = (mu_vel_diff ** 2).sum(dim=-1)
    loss_mu_vel_phase2 = (mu_vel_sum ** 2).sum(dim=-1)
    loss_mu_vel_raw = torch.min(loss_mu_vel_phase1, loss_mu_vel_phase2).mean()

    # 3. 磁跃迁偶极原始 MSE（相位不变）
    y_m = target['y_m'].reshape(batch_size, n_states, 3)
    m_diff = pred['m_total'] - y_m
    m_sum = pred['m_total'] + y_m
    loss_m_phase1 = (m_diff ** 2).sum(dim=-1)
    loss_m_phase2 = (m_sum ** 2).sum(dim=-1)
    loss_m_raw = torch.min(loss_m_phase1, loss_m_phase2).mean()

    # 4. 旋转强度原始 MSE
    y_R = target['y_R'].reshape(batch_size, n_states)
    loss_R_raw = F.mse_loss(pred['R_pred'], y_R)

    return {
        'loss_E_raw': loss_E_raw.item(),
        'loss_mu_vel_raw': loss_mu_vel_raw.item(),
        'loss_m_raw': loss_m_raw.item(),
        'loss_R_raw': loss_R_raw.item(),
    }


def train_epoch(model, loader, criterion, optimizer, device, logger):
    """训练一个 epoch，返回归一化损失和原始损失。"""
    model.train()

    total_loss = 0.0
    total_raw_losses = {'loss_E_raw': 0.0, 'loss_mu_vel_raw': 0.0, 'loss_m_raw': 0.0, 'loss_R_raw': 0.0}
    loss_components = {'loss_E': 0.0, 'loss_mu_vel': 0.0, 'loss_m': 0.0, 'loss_R': 0.0, 'loss_R_sign': 0.0}
    total_r_sign_acc = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, data in enumerate(loader):
        data = data.to(device)

        # 前向传播
        pred = model(data)

        # 准备标签字典
        target = {
            'y_E': data.y_E,
            'y_mu_vel': data.y_mu_vel,
            'y_m': data.y_m,
            'y_R': data.y_R
        }

        # 计算损失
        loss, loss_dict = criterion(pred, target)

        # 计算原始损失（用于监控）
        raw_losses = compute_raw_losses(pred, target, n_states=model.n_states)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 累积损失
        total_loss += loss_dict['loss']
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        for key in total_raw_losses:
            total_raw_losses[key] += raw_losses[key]
        total_r_sign_acc += loss_dict['R_sign_acc']
        num_batches += 1

        # 每 10 个 batch 打印一次进度
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {total_loss / num_batches:.4f} "
                f"(E: {loss_dict['loss_E']:.4f}, "
                f"mu_vel: {loss_dict['loss_mu_vel']:.4f}, "
                f"m: {loss_dict['loss_m']:.4f}, "
                f"R: {loss_dict['loss_R']:.4f}) "
                f"R_sign_acc: {loss_dict['R_sign_acc']:.4f}"
            )

    # 计算 epoch 平均值
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    for key in total_raw_losses:
        total_raw_losses[key] /= num_batches
    avg_r_sign_acc = total_r_sign_acc / num_batches

    loss_components['loss'] = avg_loss
    loss_components['R_sign_acc'] = avg_r_sign_acc
    loss_components.update(total_raw_losses)

    elapsed = time.time() - start_time
    logger.info(f"  Training epoch completed in {elapsed:.2f}s")

    return loss_components


@torch.no_grad()
def validate(model, loader, criterion, device, logger):
    """在验证集上评估模型，返回归一化损失和原始损失。"""
    model.eval()

    total_loss = 0.0
    total_raw_losses = {'loss_E_raw': 0.0, 'loss_mu_vel_raw': 0.0, 'loss_m_raw': 0.0, 'loss_R_raw': 0.0}
    loss_components = {'loss_E': 0.0, 'loss_mu_vel': 0.0, 'loss_m': 0.0, 'loss_R': 0.0, 'loss_R_sign': 0.0}
    total_r_sign_acc = 0.0
    num_batches = 0

    for data in loader:
        data = data.to(device)

        pred = model(data)

        target = {
            'y_E': data.y_E,
            'y_mu_vel': data.y_mu_vel,
            'y_m': data.y_m,
            'y_R': data.y_R
        }

        loss, loss_dict = criterion(pred, target)
        raw_losses = compute_raw_losses(pred, target, n_states=model.n_states)

        total_loss += loss_dict['loss']
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        for key in total_raw_losses:
            total_raw_losses[key] += raw_losses[key]
        total_r_sign_acc += loss_dict['R_sign_acc']
        num_batches += 1

    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    for key in total_raw_losses:
        total_raw_losses[key] /= num_batches
    avg_r_sign_acc = total_r_sign_acc / num_batches

    loss_components['loss'] = avg_loss
    loss_components['R_sign_acc'] = avg_r_sign_acc
    loss_components.update(total_raw_losses)

    return loss_components


def is_pareto_dominated(candidate, existing):
    """
    判断 candidate 是否被 existing 支配。
    帕累托目标：R_mae（R 的 L1/MAE）和 E_mse（E 的 MSE），均越小越好。
    两个目标直接对应 ECD 光谱预测质量，不引入 μ_vel/m 等中间量。
    """
    objectives = ['R_mae', 'E_mse']  # 均越小越好

    not_worse      = all(existing[k] <= candidate[k] for k in objectives)
    strictly_better = any(existing[k] <  candidate[k] for k in objectives)

    return not_worse and strictly_better


def update_pareto_frontier(pareto_checkpoints, candidate_loss, candidate_info, max_checkpoints=10):
    """
    将新候选加入帕累托前沿（若未被支配），并剔除被其支配的旧检查点。
    返回更新后的前沿列表和需要删除的 checkpoint 文件路径。
    """
    # 如果被任一现有点支配，直接拒绝
    for existing_loss, _ in pareto_checkpoints:
        if is_pareto_dominated(candidate_loss, existing_loss):
            return pareto_checkpoints, []

    # 加入前沿，同时移除被新点支配的旧点
    new_pareto = []
    removed_paths = []
    for existing_loss, existing_info in pareto_checkpoints:
        if not is_pareto_dominated(existing_loss, candidate_loss):
            new_pareto.append((existing_loss, existing_info))
        else:
            if 'checkpoint_path' in existing_info:
                removed_paths.append(existing_info['checkpoint_path'])
    new_pareto.append((candidate_loss, candidate_info))

    # 超出上限时按综合评分裁剪：R_mae 优先（权重更高），E_mse 次之
    if len(new_pareto) > max_checkpoints:
        def pareto_score(item):
            d = item[0]
            return d.get('R_mae', 0) * 10 + d.get('E_mse', 0)
        new_pareto.sort(key=pareto_score)
        for _, info in new_pareto[max_checkpoints:]:
            if 'checkpoint_path' in info:
                removed_paths.append(info['checkpoint_path'])
        new_pareto = new_pareto[:max_checkpoints]

    return new_pareto, removed_paths


def save_pareto_checkpoint(model, optimizer, scheduler, epoch, val_components,
                           checkpoint_dir, config, pareto_checkpoints, logger,
                           max_checkpoints=10):
    """
    基于帕累托最优策略保存检查点。
    帕累托目标：R_mae（验证集 R 的 L1/MAE）和 E_mse（验证集 E 的 MSE），均越小越好。
    这两个指标直接对应 ECD 光谱预测质量，μ_vel/m 作为训练辅助量不纳入选模型标准。
    """
    candidate_loss = {
        'R_mae': val_components.get('loss_R', float('inf')),    # F.l1_loss(R_pred, R_true) = R MAE
        'E_mse': val_components.get('loss_E_raw', float('inf')), # F.mse_loss(E_pred, E_true)
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'pareto_epoch_{epoch}.pt')
    candidate_info  = {
        'epoch': epoch,
        'checkpoint_path': checkpoint_path,
        'total_normalized_loss': val_components.get('loss', float('inf')),
    }

    # 先写文件（update_pareto_frontier 可能删除被支配的旧文件）
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss':             val_components.get('loss', float('inf')),
        'pareto_metrics':       candidate_loss,
        'config':               config,
    }, checkpoint_path)

    pareto_checkpoints, removed_paths = update_pareto_frontier(
        pareto_checkpoints, candidate_loss, candidate_info, max_checkpoints
    )

    for path in removed_paths:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"  Removed dominated checkpoint: {os.path.basename(path)}")

    # 若新点未进入前沿（仍被支配），删除刚才写的文件
    in_frontier = any(info.get('checkpoint_path') == checkpoint_path
                      for _, info in pareto_checkpoints)
    if not in_frontier and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    logger.info(f"  Pareto frontier: {len(pareto_checkpoints)} checkpoints")
    for i, (d, info) in enumerate(pareto_checkpoints):
        logger.info(f"    [{i+1}] Epoch {info['epoch']}: "
                    f"R_mae={d['R_mae']:.4f} (10^-40 cgs), "
                    f"E_mse={d['E_mse']:.6f} eV²")

    return pareto_checkpoints


def plot_loss_curves(train_losses, val_losses, save_path):
    """绘制并保存训练/验证损失曲线（3×3，9 个子图）。"""
    fig = plt.figure(figsize=(20, 12))

    epochs = range(1, len(train_losses['total']) + 1)

    # 第 1 行：归一化总损失 + E 归一化 + 联合 μ-m 归一化
    plt.subplot(3, 3, 1)
    plt.plot(epochs, train_losses['total'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['total'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('Total Normalized Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.subplot(3, 3, 2)
    plt.plot(epochs, train_losses['E'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['E'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('Excitation Energy Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.subplot(3, 3, 3)
    plt.plot(epochs, train_losses['norm_mu_m'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['norm_mu_m'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('Joint mu_vel + m Loss (Normalized, same phase)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 第 2 行：R 归一化 + R_sign_acc + 激发能原始 MSE
    plt.subplot(3, 3, 4)
    plt.plot(epochs, train_losses['R'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['R'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('Rotatory Strength Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.subplot(3, 3, 5)
    plt.plot(epochs, train_losses['R_sign_acc'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['R_sign_acc'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('R Sign Prediction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.subplot(3, 3, 6)
    plt.plot(epochs, train_losses['E_raw'], 'b-', label='Train (Raw)', linewidth=2)
    plt.plot(epochs, val_losses['E_raw'], 'r-', label='Val (Raw)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Excitation Energy Loss (Raw MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 第 3 行：速度电偶极矩、磁偶极矩、旋转强度的原始 MSE
    plt.subplot(3, 3, 7)
    plt.plot(epochs, train_losses['mu_vel_raw'], 'b-', label='Train (Raw)', linewidth=2)
    plt.plot(epochs, val_losses['mu_vel_raw'], 'r-', label='Val (Raw)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Velocity Electric Dipole Loss (Raw MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.subplot(3, 3, 8)
    plt.plot(epochs, train_losses['m_raw'], 'b-', label='Train (Raw)', linewidth=2)
    plt.plot(epochs, val_losses['m_raw'], 'r-', label='Val (Raw)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Magnetic Dipole Loss (Raw MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.subplot(3, 3, 9)
    plt.plot(epochs, train_losses['R_raw'], 'b-', label='Train (Raw)', linewidth=2)
    plt.plot(epochs, val_losses['R_raw'], 'r-', label='Val (Raw)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Rotatory Strength Loss (Raw MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """主训练函数。"""
    config = {
        'data_dir': 'data/processed_with_enantiomers',
        'checkpoint_dir': 'checkpoints',
        'batch_size': 64,
        'num_epochs': 1000,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_workers': 16,
        # 模型超参数
        'num_features': 128,
        'max_l': 3,
        'num_blocks': 3,
        'num_radial': 32,
        'cutoff': 50.0,
        'n_states': 20,
        'max_atomic_number': 60,
        # 损失权重（EMA 归一化后的相对权重）
        'lambda_E': 1.0,
        'lambda_mu_vel': 1.0,   # 联合 μ-m 相位损失权重（同时约束 μ 和 m）
        'lambda_m': 1.0,        # 保留参数，mu-m 联合约束中已隐含，不再单独加权
        'lambda_R': 1.0,
        'lambda_R_sign': 0.0,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(config['checkpoint_dir'])

    logger.info("=" * 80)
    logger.info("PhysECD Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Configuration: {config}")

    # 加载数据
    logger.info("\nLoading data...")
    train_loader, val_loader = load_data(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # 初始化模型
    logger.info("\nInitializing model...")
    model = PhysECDModel(
        num_features=config['num_features'],
        max_l=config['max_l'],
        num_blocks=config['num_blocks'],
        num_radial=config['num_radial'],
        cutoff=config['cutoff'],
        n_states=config['n_states'],
        max_atomic_number=config['max_atomic_number']
    ).to(device)

    num_params = model.get_num_params()
    logger.info(f"Model initialized with {num_params:,} trainable parameters")

    # 初始化损失函数
    criterion = PhysECDLoss(
        lambda_E=config['lambda_E'],
        lambda_mu_vel=config['lambda_mu_vel'],
        lambda_m=config['lambda_m'],
        lambda_R=config['lambda_R'],
        lambda_R_sign=config['lambda_R_sign']
    ).to(device)

    # 初始化优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 余弦退火：每 num_epochs/5 个 epoch 为一个完整周期
    scheduler = CosineAnnealingLR(
        optimizer,
        eta_min=1e-5,
        T_max=config['num_epochs'] // 5
    )

    # 训练记录
    best_val_loss = float('inf')
    pareto_checkpoints = []   # 帕累托前沿检查点列表
    train_losses = {
        'total': [], 'E': [], 'norm_mu_m': [], 'R': [], 'R_sign_acc': [],
        'E_raw': [], 'mu_vel_raw': [], 'm_raw': [], 'R_raw': []
    }
    val_losses = {
        'total': [], 'E': [], 'norm_mu_m': [], 'R': [], 'R_sign_acc': [],
        'E_raw': [], 'mu_vel_raw': [], 'm_raw': [], 'R_raw': []
    }

    # 训练循环
    logger.info("\nStarting training...")
    logger.info("=" * 80)

    for epoch in range(1, config['num_epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['num_epochs']}")
        logger.info("-" * 80)

        # 训练一个 epoch
        train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )

        # 验证
        logger.info("  Running validation...")
        val_components = validate(
            model, val_loader, criterion, device, logger
        )

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 打印 epoch 总结
        logger.info("-" * 80)
        logger.info(
            f"Epoch {epoch} Summary - "
            f"Train Loss: {train_components['loss']:.4f}, "
            f"Val Loss: {val_components['loss']:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        logger.info(
            f"  Raw MSE - Train: E={train_components['loss_E_raw']:.4f}, "
            f"mu_vel={train_components['loss_mu_vel_raw']:.4f}, "
            f"m={train_components['loss_m_raw']:.4f}, "
            f"R={train_components['loss_R_raw']:.4f}"
        )
        logger.info(
            f"  Raw MSE - Val:   E={val_components['loss_E_raw']:.4f}, "
            f"mu_vel={val_components['loss_mu_vel_raw']:.4f}, "
            f"m={val_components['loss_m_raw']:.4f}, "
            f"R={val_components['loss_R_raw']:.4f}"
        )
        logger.info(
            f"  R Sign Acc: Train={train_components['R_sign_acc']:.4f}, "
            f"Val={val_components['R_sign_acc']:.4f}"
        )

        # 记录损失曲线数据
        key_map = {
            'total': 'loss', 'E': 'loss_E', 'norm_mu_m': 'norm_mu_m',
            'R': 'loss_R', 'R_sign_acc': 'R_sign_acc',
            'E_raw': 'loss_E_raw', 'mu_vel_raw': 'loss_mu_vel_raw',
            'm_raw': 'loss_m_raw', 'R_raw': 'loss_R_raw'
        }
        for plot_key, comp_key in key_map.items():
            if comp_key in train_components:
                train_losses[plot_key].append(train_components[comp_key])
            if comp_key in val_components:
                val_losses[plot_key].append(val_components[comp_key])

        # 保存最佳模型（按验证集 R MAE，直接对应 ECD 光谱预测质量）
        if val_components['loss_R'] < best_val_loss:
            best_val_loss = val_components['loss_R']
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_components['loss'],
                'config': config
            }, checkpoint_path)
            logger.info(f"  Saved best model (val R MAE: {val_components['loss_R']:.4f})")

        # 帕累托前沿检查点（四目标：E/mu_vel/m raw MSE + R_sign_acc）
        pareto_checkpoints = save_pareto_checkpoint(
            model, optimizer, scheduler, epoch, val_components,
            config['checkpoint_dir'], config, pareto_checkpoints, logger,
            max_checkpoints=10
        )

        # 每 100 epoch 保存一次 checkpoint
        if epoch % 100 == 0:
            checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_components['loss'],
                'config': config
            }, checkpoint_path)
            logger.info(f"  Saved checkpoint at epoch {epoch}")

        # 每 5 epoch 更新一次损失曲线图
        if epoch % 5 == 0 or epoch == 1:
            plot_path = os.path.join(config['checkpoint_dir'], 'loss_curves.png')
            plot_loss_curves(train_losses, val_losses, plot_path)
            logger.info(f"  Updated loss curves plot")

    # 训练完成
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    logger.info(f"Best val R MAE: {best_val_loss:.4f} (10^-40 cgs)")
    logger.info(f"Best model saved to: {os.path.join(config['checkpoint_dir'], 'best_model.pt')}")
    logger.info("=" * 80)

    # 保存最终损失曲线图
    plot_path = os.path.join(config['checkpoint_dir'], 'loss_curves_final.png')
    plot_loss_curves(train_losses, val_losses, plot_path)


if __name__ == '__main__':
    main()
