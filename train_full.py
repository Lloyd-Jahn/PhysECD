"""
完整训练验证测试脚本

基于过拟合测试通过后的模型，进行全量数据训练和测试。
训练完成后生成测试集前100个分子的光谱对比图。
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physecd.models import PhysECDModel
from physecd.physics import PhysECDLoss


def setup_logging(log_dir):
    """Setup logging configuration."""
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


def load_data(data_dir, batch_size=64, num_workers=4):
    """加载完整的训练、验证和测试数据集。"""
    logger = logging.getLogger(__name__)

    train_path = os.path.join(data_dir, 'train.pt')
    val_path = os.path.join(data_dir, 'val.pt')
    test_path = os.path.join(data_dir, 'test.pt')

    logger.info(f"Loading training data from {train_path}")
    train_data = torch.load(train_path, weights_only=False)
    logger.info(f"Loaded {len(train_data)} training samples")

    logger.info(f"Loading validation data from {val_path}")
    val_data = torch.load(val_path, weights_only=False)
    logger.info(f"Loaded {len(val_data)} validation samples")

    logger.info(f"Loading test data from {test_path}")
    test_data = torch.load(test_path, weights_only=False)
    logger.info(f"Loaded {len(test_data)} test samples")

    # Create data loaders
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

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def compute_raw_losses(pred, target, n_states=20):
    """
    计算原始损失（未归一化的 MSE/L1）。
    
    Returns:
        dict: 包含原始损失的字典，键名为 raw_loss_E, raw_loss_mu 等
               以及 R_sign_acc 用于帕累托前沿
    """
    import torch.nn.functional as F
    
    batch_size = pred['E_pred'].shape[0]
    
    # 1. 激发能 Loss
    y_E = target['y_E'].reshape(batch_size, n_states)
    raw_loss_E = F.mse_loss(pred['E_pred'], y_E)
    
    # 2. 电跃迁偶极 Loss - Phase-Invariant
    y_mu = target['y_mu_vel'].reshape(batch_size, n_states, 3)
    mu_diff = pred['mu_total'] - y_mu
    mu_sum = pred['mu_total'] + y_mu
    loss_mu_phase1 = (mu_diff ** 2).sum(dim=-1)
    loss_mu_phase2 = (mu_sum ** 2).sum(dim=-1)
    raw_loss_mu = torch.min(loss_mu_phase1, loss_mu_phase2).mean()
    
    # 3. 磁跃迁偶极 Loss - Phase-Invariant
    y_m = target['y_m'].reshape(batch_size, n_states, 3)
    m_diff = pred['m_total'] - y_m
    m_sum = pred['m_total'] + y_m
    loss_m_phase1 = (m_diff ** 2).sum(dim=-1)
    loss_m_phase2 = (m_sum ** 2).sum(dim=-1)
    raw_loss_m = torch.min(loss_m_phase1, loss_m_phase2).mean()
    
    # 4. 旋转强度 R - 使用 Sign Accuracy 而不是 L1 loss
    y_R = target['y_R'].reshape(batch_size, n_states)
    y_R_sign = (y_R > 0).float()
    R_sign_pred = (pred['R_pred'] > 0).float()
    R_sign_acc = (R_sign_pred == y_R_sign).float().mean()
    
    return {
        'raw_loss_E': raw_loss_E.item(),
        'raw_loss_mu': raw_loss_mu.item(),
        'raw_loss_m': raw_loss_m.item(),
        'R_sign_acc': R_sign_acc.item(),  # 使用 sign accuracy 而不是 raw_loss_R
    }


def train_epoch(model, loader, criterion, optimizer, device, logger):
    """训练一个 epoch。"""
    model.train()

    # 使用字典来累积所有损失，提高扩展性
    accumulated_losses = {}
    num_batches = 0

    start_time = time.time()

    for batch_idx, data in enumerate(loader):
        data = data.to(device)

        # Forward pass
        pred = model(data)

        # Prepare target dictionary
        target = {
            'y_E': data.y_E,
            'y_mu_vel': data.y_mu_vel,
            'y_m': data.y_m,
            'y_R': data.y_R
        }

        # Compute loss
        loss, loss_dict = criterion(pred, target)
        
        # Compute raw losses (用于帕累托最优)
        raw_losses = compute_raw_losses(pred, target, n_states=model.n_states)
        # 合并到 loss_dict
        loss_dict.update(raw_losses)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses (动态处理所有键)
        for key, value in loss_dict.items():
            if key not in accumulated_losses:
                accumulated_losses[key] = 0.0
            accumulated_losses[key] += value
        num_batches += 1

        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            # 动态构建日志字符串
            loss_parts = [f"{k}: {v:.4f}" for k, v in loss_dict.items() 
                         if k.startswith('loss_') and not k.startswith('loss_R_sign')][:5]
            logger.info(
                f"  Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss_dict['loss']:.4f} "
                f"({', '.join(loss_parts)}) "
                f"R_sign_acc: {loss_dict.get('R_sign_acc', 0):.4f}"
            )

    # Compute epoch averages
    avg_losses = {key: value / num_batches for key, value in accumulated_losses.items()}

    elapsed = time.time() - start_time
    logger.info(f"  Training epoch completed in {elapsed:.2f}s")

    return avg_losses


@torch.no_grad()
def validate(model, loader, criterion, device, logger):
    """验证模型。"""
    model.eval()

    # 使用字典来累积所有损失
    accumulated_losses = {}
    num_batches = 0

    for data in loader:
        data = data.to(device)

        # Forward pass
        pred = model(data)

        # Prepare target dictionary
        target = {
            'y_E': data.y_E,
            'y_mu_vel': data.y_mu_vel,
            'y_m': data.y_m,
            'y_R': data.y_R
        }

        # Compute loss
        loss, loss_dict = criterion(pred, target)
        
        # Compute raw losses (用于帕累托最优)
        raw_losses = compute_raw_losses(pred, target, n_states=model.n_states)
        # 合并到 loss_dict
        loss_dict.update(raw_losses)

        # Accumulate losses (动态处理所有键)
        for key, value in loss_dict.items():
            if key not in accumulated_losses:
                accumulated_losses[key] = 0.0
            accumulated_losses[key] += value
        num_batches += 1

    # Compute averages
    avg_losses = {key: value / num_batches for key, value in accumulated_losses.items()}

    return avg_losses


@torch.no_grad()
def test(model, loader, criterion, device, logger):
    """测试模型。"""
    logger.info("\nRunning test evaluation...")
    return validate(model, loader, criterion, device, logger)


def is_pareto_dominated(candidate, existing):
    """
    检查 candidate 是否被 existing 支配。
    注意：R_sign_acc 是越大越好，而其他损失是越小越好
    
    Args:
        candidate: dict with keys 'raw_loss_E', 'raw_loss_mu', 'raw_loss_m', 'R_sign_acc'
        existing: dict with same keys
    
    Returns:
        bool: True if candidate is dominated by existing
    """
    # 损失目标 (越小越好)
    loss_objectives = ['raw_loss_E', 'raw_loss_mu', 'raw_loss_m']
    # 准确率目标 (越大越好)
    acc_objectives = ['R_sign_acc']
    
    # existing 在所有损失目标上都不比 candidate 差 (existing <= candidate)
    loss_not_worse = all(existing[obj] <= candidate[obj] for obj in loss_objectives)
    # existing 在所有准确率目标上都不比 candidate 差 (existing >= candidate)
    acc_not_worse = all(existing[obj] >= candidate[obj] for obj in acc_objectives)
    
    # existing 至少在一个目标上更好
    loss_strictly_better = any(existing[obj] < candidate[obj] for obj in loss_objectives)
    acc_strictly_better = any(existing[obj] > candidate[obj] for obj in acc_objectives)
    
    not_worse = loss_not_worse and acc_not_worse
    strictly_better = loss_strictly_better or acc_strictly_better
    
    return not_worse and strictly_better


def update_pareto_frontier(pareto_checkpoints, candidate_loss, candidate_info, max_checkpoints=10):
    """
    更新帕累托前沿，添加新的检查点如果被接受。
    注意：R_sign_acc 越大越好，而损失越小越好
    
    Args:
        pareto_checkpoints: list of (loss_dict, info_dict) tuples
        candidate_loss: dict with raw losses and R_sign_acc
        candidate_info: dict with checkpoint info (epoch, paths, etc.)
        max_checkpoints: maximum number of checkpoints to keep
    
    Returns:
        updated list of pareto checkpoints, list of removed checkpoint paths to delete
    """
    # 检查 candidate 是否被任何现有检查点支配
    for existing_loss, _ in pareto_checkpoints:
        if is_pareto_dominated(candidate_loss, existing_loss):
            # Candidate 被支配，不添加
            return pareto_checkpoints, []
    
    # Candidate 不被支配，添加到前沿
    new_pareto = []
    removed_paths = []
    
    for existing_loss, existing_info in pareto_checkpoints:
        if not is_pareto_dominated(existing_loss, candidate_loss):
            # Existing 不被 candidate 支配，保留
            new_pareto.append((existing_loss, existing_info))
        else:
            # Existing 被 candidate 支配，标记删除
            if 'checkpoint_path' in existing_info:
                removed_paths.append(existing_info['checkpoint_path'])
    
    # 添加新的检查点
    new_pareto.append((candidate_loss, candidate_info))
    
    # 如果检查点太多，保留最好的 (基于综合评分：损失越低越好，sign_acc越高越好)
    if len(new_pareto) > max_checkpoints:
        # 计算每个检查点的综合评分
        def pareto_score(item):
            loss_dict = item[0]
            # 损失的加权和 (越小越好)
            loss_sum = (loss_dict.get('raw_loss_E', 0) + 
                       loss_dict.get('raw_loss_mu', 0) + 
                       loss_dict.get('raw_loss_m', 0))
            # sign_acc (越大越好，用 1 - acc 使其成为越小越好)
            sign_penalty = 1.0 - loss_dict.get('R_sign_acc', 0)
            return loss_sum + sign_penalty * 10  # sign_acc 权重为10
        
        new_pareto.sort(key=pareto_score)
        # 标记要删除的多余检查点
        for i in range(max_checkpoints, len(new_pareto)):
            _, info = new_pareto[i]
            if 'checkpoint_path' in info:
                removed_paths.append(info['checkpoint_path'])
        new_pareto = new_pareto[:max_checkpoints]
    
    return new_pareto, removed_paths


def save_pareto_checkpoint(model, optimizer, scheduler, epoch, val_losses, checkpoint_dir, pareto_checkpoints, logger, max_checkpoints=10):
    """
    基于帕累托最优策略保存检查点。
    使用 E, mu, m 的原始损失和 R 的 sign accuracy 作为帕累托目标。
    
    Args:
        val_losses: dict containing validation losses including raw losses and R_sign_acc
        pareto_checkpoints: list to maintain pareto frontier
    
    Returns:
        updated pareto_checkpoints list
    """
    # 准备 candidate 信息 (使用 E, mu, m 的损失和 R 的 sign accuracy)
    candidate_loss = {
        'raw_loss_E': val_losses.get('raw_loss_E', float('inf')),
        'raw_loss_mu': val_losses.get('raw_loss_mu', float('inf')),
        'raw_loss_m': val_losses.get('raw_loss_m', float('inf')),
        'R_sign_acc': val_losses.get('R_sign_acc', 0.0),  # 使用 sign accuracy 而不是 raw_loss_R
    }
    
    checkpoint_filename = f'pareto_epoch_{epoch}.pt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    candidate_info = {
        'epoch': epoch,
        'checkpoint_path': checkpoint_path,
        'total_normalized_loss': val_losses.get('loss', float('inf')),
    }
    
    # 先保存检查点文件
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_losses.get('loss', float('inf')),
        'pareto_metrics': candidate_loss,  # 保存帕累托指标
        'config': None,  # Will be filled by caller
    }, checkpoint_path)
    
    # 更新帕累托前沿
    pareto_checkpoints, removed_paths = update_pareto_frontier(
        pareto_checkpoints, candidate_loss, candidate_info, max_checkpoints
    )
    
    # 删除被支配的检查点文件
    for path in removed_paths:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"  Removed dominated checkpoint: {os.path.basename(path)}")
    
    # 记录帕累托前沿状态
    logger.info(f"  Pareto frontier: {len(pareto_checkpoints)} checkpoints")
    for i, (loss_dict, info) in enumerate(pareto_checkpoints):
        logger.info(f"    [{i+1}] Epoch {info['epoch']}: "
                   f"E={loss_dict['raw_loss_E']:.4f}, "
                   f"mu={loss_dict['raw_loss_mu']:.4f}, "
                   f"m={loss_dict['raw_loss_m']:.4f}, "
                   f"R_sign_acc={loss_dict['R_sign_acc']:.4f}")
    
    return pareto_checkpoints


def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot and save training/validation loss curves dynamically based on loss_dict keys."""
    logger = logging.getLogger(__name__)
    
    # 获取所有可用的 loss keys（排除空列表的）
    all_keys = set()
    for key, values in train_losses.items():
        if values and len(values) > 0:
            all_keys.add(key)
    for key, values in val_losses.items():
        if values and len(values) > 0:
            all_keys.add(key)
    
    if not all_keys:
        logger.warning("No loss data to plot yet")
        return
    
    # 排序 keys：total loss 在前，然后是 raw losses，然后是 normalized losses，最后是 metrics
    priority_order = {
        'loss': 0,  # Total normalized loss
        'raw_loss_E': 1,
        'raw_loss_mu': 2,
        'raw_loss_m': 3,
        'raw_loss_R': 4,
        'norm_E': 5,
        'norm_mu': 6,
        'norm_m': 7,
        'norm_mu_m': 8,
        'norm_R': 9,
        'R_sign_acc': 10,
    }
    sorted_keys = sorted(all_keys, key=lambda k: priority_order.get(k, 99))
    
    # 获取 epoch 数
    epochs = range(1, len(train_losses.get('loss', train_losses.get(sorted_keys[0], []))) + 1)
    if len(epochs) == 0:
        logger.warning("No loss data to plot yet")
        return
    
    # 根据 key 数量动态计算 subplot 布局
    n_keys = len(sorted_keys)
    n_cols = 3
    n_rows = (n_keys + n_cols - 1) // n_cols  # 向上取整
    n_rows = max(n_rows, 2)  # 至少 2 行
    
    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    
    # 定义每个 key 的显示配置
    key_configs = {
        'loss': {'title': 'Total Normalized Loss', 'ylabel': 'Normalized Loss', 'yscale': 'log'},
        'raw_loss_E': {'title': 'Excitation Energy Loss (Raw)', 'ylabel': 'MSE Loss', 'yscale': 'log'},
        'raw_loss_mu': {'title': 'Electric Dipole Loss (Raw)', 'ylabel': 'MSE Loss', 'yscale': 'log'},
        'raw_loss_m': {'title': 'Magnetic Dipole Loss (Raw)', 'ylabel': 'MSE Loss', 'yscale': 'log'},
        'raw_loss_R': {'title': 'Rotatory Strength Loss (Raw L1)', 'ylabel': 'L1 Loss', 'yscale': 'log'},
        'norm_E': {'title': 'Energy Loss (Normalized)', 'ylabel': 'Normalized Loss', 'yscale': 'log'},
        'norm_mu': {'title': 'Electric Dipole Loss (Normalized)', 'ylabel': 'Normalized Loss', 'yscale': 'log'},
        'norm_m': {'title': 'Magnetic Dipole Loss (Normalized)', 'ylabel': 'Normalized Loss', 'yscale': 'log'},
        'norm_mu_m': {'title': 'Mu-M Joint Loss (Normalized)', 'ylabel': 'Normalized Loss', 'yscale': 'log'},
        'norm_R': {'title': 'Rotatory Strength Loss (Normalized)', 'ylabel': 'Normalized Loss', 'yscale': 'log'},
        'R_sign_acc': {'title': 'R Sign Prediction Accuracy', 'ylabel': 'Accuracy', 'ylim': [0, 1]},
    }
    
    for idx, key in enumerate(sorted_keys, 1):
        plt.subplot(n_rows, n_cols, idx)
        
        train_vals = train_losses.get(key, [])
        val_vals = val_losses.get(key, [])
        
        if train_vals:
            plt.plot(epochs[:len(train_vals)], train_vals, 'b-', label='Train', linewidth=2)
        if val_vals:
            plt.plot(epochs[:len(val_vals)], val_vals, 'r-', label='Validation', linewidth=2)
        
        config = key_configs.get(key, {'title': key.replace('_', ' ').title(), 'ylabel': 'Value'})
        plt.xlabel('Epoch')
        plt.ylabel(config['ylabel'])
        plt.title(config['title'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if 'yscale' in config:
            plt.yscale(config['yscale'])
        if 'ylim' in config:
            plt.ylim(config['ylim'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved loss curves to {save_path}")


def generate_spectrum(E_pred, R_pred_au, wavelength_grid, sigma=0.4):
    """
    应用高斯展宽生成连续 ECD 光谱。
    
    物理公式:
    1. R_cgs = R_au × 471.44 (in 10^-40 cgs units)
    2. E_grid = 1240 / λ (eV)
    3. Δε(E) = (1 / (2.296×10^1 × σ × √π)) × Σ E_i × R_cgs,i × exp[-(E - E_i)^2 / σ^2]
    4. [θ] = Δε × 3298.2
    """
    R_cgs = R_pred_au * 471.44
    E_grid = 1240.0 / wavelength_grid
    norm_constant = 2.296e1 * sigma * np.sqrt(np.pi)
    delta_epsilon = np.zeros_like(E_grid)
    
    for i in range(len(E_pred)):
        gaussian = np.exp(-((E_grid - E_pred[i]) / sigma) ** 2)
        delta_epsilon += E_pred[i] * R_cgs[i] * gaussian
    
    delta_epsilon /= norm_constant
    molar_ellipticity = delta_epsilon * 3298.2
    
    return molar_ellipticity


def plot_spectrum_comparison(wavelength, pred_spectrum, real_spectrum, mol_id, 
                              save_path, title=None):
    """绘制预测光谱和真实光谱的对比图。"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    ax.plot(wavelength, real_spectrum, 'b-', linewidth=2.0, label='Real', alpha=0.8)
    ax.plot(wavelength, pred_spectrum, 'r--', linewidth=2.0, label='Predicted', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('[θ] (deg·cm²/dmol)', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title(f'Molecule {mol_id}: ECD Spectrum Comparison', fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlim(wavelength.min(), wavelength.max())
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_test_spectra(model, test_loader, device, output_dir, logger, num_samples=100):
    """
    为测试集前 num_samples 个分子生成并绘制光谱对比图。
    """
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    wavelength_min, wavelength_max, wavelength_step = 80.0, 450.0, 1.0
    wavelength_grid = np.arange(wavelength_min, wavelength_max + wavelength_step, wavelength_step)
    sigma = 0.4
    
    logger.info(f"\nGenerating spectra for first {num_samples} test molecules...")
    logger.info(f"Spectra will be saved to: {output_dir}")
    
    count = 0
    all_spectrum_errors = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if count >= num_samples:
                break
                
            data = data.to(device)
            pred = model(data)
            
            batch_size = data.num_graphs
            n_states = model.n_states
            
            for i in range(batch_size):
                if count >= num_samples:
                    break
                
                # 获取预测值
                E_pred = pred['E_pred'][i].cpu().numpy()
                R_pred_au = pred['R_pred'][i].cpu().numpy()
                
                # 获取真实值
                if data.y_E.dim() == 2:
                    E_real = data.y_E[i].cpu().numpy()
                    R_real_cgs = data.y_R[i].cpu().numpy()
                else:
                    start_idx = i * n_states
                    end_idx = start_idx + n_states
                    E_real = data.y_E[start_idx:end_idx].cpu().numpy()
                    R_real_cgs = data.y_R[start_idx:end_idx].cpu().numpy()
                
                # 将真实 R 从 cgs 转换为原子单位
                R_real_au = R_real_cgs / 471.44
                R_pred_au = R_pred_au / 471.44
                
                # 生成光谱
                pred_spectrum = generate_spectrum(E_pred, R_pred_au, wavelength_grid, sigma)
                real_spectrum = generate_spectrum(E_real, R_real_au, wavelength_grid, sigma)
                
                # 获取分子 ID
                mol_id = data.mol_id[i].item() if hasattr(data, 'mol_id') else count
                
                # 绘制对比图
                save_path = os.path.join(output_dir, f'spectrum_mol_{mol_id}.png')
                plot_spectrum_comparison(wavelength_grid, pred_spectrum, real_spectrum, 
                                        mol_id, save_path)
                
                # 计算光谱差异
                spectrum_diff = np.mean(np.abs(pred_spectrum - real_spectrum))
                all_spectrum_errors.append(spectrum_diff)
                
                count += 1
                
                if count % 10 == 0:
                    logger.info(f"  Generated {count}/{num_samples} spectra...")
    
    # 计算并输出统计信息
    mean_error = np.mean(all_spectrum_errors)
    std_error = np.std(all_spectrum_errors)
    median_error = np.median(all_spectrum_errors)
    
    logger.info(f"\nSpectrum Generation Complete!")
    logger.info(f"  Total spectra generated: {count}")
    logger.info(f"  Mean absolute difference: {mean_error:.4e}")
    logger.info(f"  Median absolute difference: {median_error:.4e}")
    logger.info(f"  Std of differences: {std_error:.4e}")
    
    # 保存统计信息到文件
    stats_path = os.path.join(output_dir, 'spectrum_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Spectrum Comparison Statistics\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Total spectra generated: {count}\n")
        f.write(f"Mean absolute difference: {mean_error:.4e}\n")
        f.write(f"Median absolute difference: {median_error:.4e}\n")
        f.write(f"Std of differences: {std_error:.4e}\n")
        f.write(f"\nPer-molecule errors:\n")
        for i, err in enumerate(all_spectrum_errors):
            f.write(f"  Molecule {i}: {err:.4e}\n")
    
    logger.info(f"  Statistics saved to: {stats_path}")


def main():
    """主训练函数。"""
    # Configuration - 完全照搬 03_train.py
    config = {
        'data_dir': 'data/processed_with_enantiomers',
        'checkpoint_dir': 'checkpoints_new',
        'spectrum_dir': 'test_spectra',  # 专门存放光谱图的文件夹
        'batch_size': 64,
        'num_epochs': 1000,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_workers': 16,
        # Model hyperparameters
        'num_features': 128,
        'max_l': 3,
        'num_blocks': 3,
        'num_radial': 32,
        'cutoff': 50.0,
        'n_states': 20,
        'max_atomic_number': 60,
        
        # Loss Weights（归一化后的相对权重）
        'lambda_E': 1.0,
        'lambda_mu': 1.0,
        'lambda_m': 1.0,
        'lambda_R': 1.0,
        'lambda_R_sign': 0.0,
        
        # 光谱生成参数
        'num_test_spectra': 100,  # 生成测试集前100个分子的光谱
    }

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(config['checkpoint_dir'])

    logger.info("=" * 80)
    logger.info("PhysECD Full Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Configuration: {config}")

    # Load data
    logger.info("\nLoading data...")
    train_loader, val_loader, test_loader = load_data(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Initialize model
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

    # Initialize loss function
    criterion = PhysECDLoss(
        lambda_E=config['lambda_E'],
        lambda_mu=config['lambda_mu'],
        lambda_m=config['lambda_m'],
        lambda_R=config['lambda_R'],
        lambda_R_sign=config['lambda_R_sign']
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        eta_min=1e-5,
        T_max=config['num_epochs']/25
    )

    # Training tracking - 使用动态字典存储所有 losses
    train_losses_history = {}  # 存储所有训练 loss
    val_losses_history = {}    # 存储所有验证 loss
    pareto_checkpoints = []    # 帕累托前沿检查点列表

    # Training loop
    logger.info("\nStarting training...")
    logger.info("=" * 80)

    for epoch in range(1, config['num_epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['num_epochs']}")
        logger.info("-" * 80)

        # Train
        train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )

        # Validate
        logger.info("  Running validation...")
        val_components = validate(
            model, val_loader, criterion, device, logger
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log epoch summary (动态格式)
        logger.info("-" * 80)
        logger.info(
            f"Epoch {epoch} Summary - "
            f"Train Loss: {train_components.get('loss', 0):.4f}, "
            f"Val Loss: {val_components.get('loss', 0):.4f}, "
            f"LR: {current_lr:.6f}"
        )
        
        # 动态构建 Raw Loss 日志
        raw_loss_keys = ['raw_loss_E', 'raw_loss_mu', 'raw_loss_m', 'raw_loss_R', 'raw_loss_mu_m_joint']
        raw_train_parts = []
        raw_val_parts = []
        for key in raw_loss_keys:
            if key in train_components:
                loss_name = key.replace('raw_loss_', '')
                raw_train_parts.append(f"{loss_name}={train_components[key]:.4f}")
                raw_val_parts.append(f"{loss_name}={val_components[key]:.4f}")
        
        if raw_train_parts:
            logger.info(f"  Raw - Train: {', '.join(raw_train_parts)}")
            logger.info(f"  Raw - Val:   {', '.join(raw_val_parts)}")
        
        # R Sign Accuracy
        if 'R_sign_acc' in train_components:
            logger.info(
                f"  R Sign Acc: Train={train_components['R_sign_acc']:.4f}, "
                f"Val={val_components['R_sign_acc']:.4f}"
            )

        # Store losses for plotting (动态存储)
        for key, value in train_components.items():
            if key not in train_losses_history:
                train_losses_history[key] = []
            train_losses_history[key].append(value)
        
        for key, value in val_components.items():
            if key not in val_losses_history:
                val_losses_history[key] = []
            val_losses_history[key].append(value)

        # Save checkpoints using Pareto optimal strategy (基于未归一化的 raw losses)
        pareto_checkpoints = save_pareto_checkpoint(
            model, optimizer, scheduler, epoch, val_components,
            config['checkpoint_dir'], pareto_checkpoints, logger,
            max_checkpoints=10
        )

        # Plot loss curves
        if epoch % 5 == 0 or epoch == 1:
            plot_path = os.path.join(config['checkpoint_dir'], 'loss_curves.png')
            plot_loss_curves(train_losses_history, val_losses_history, plot_path)
            logger.info(f"  Updated loss curves plot")
    
    # 加载训练好的模型进行测试
    # last_checkpoint = torch.load('checkpoints/checkpoint_epoch_1000.pt')
    # model.load_state_dict(last_checkpoint['model_state_dict'])
    # logger.info(f"Loaded checkpoint from epoch {last_checkpoint['epoch']}")
    
    # Final test evaluation
    logger.info("\n" + "=" * 80)
    logger.info("Final Test Evaluation")
    logger.info("=" * 80)
    test_components = test(model, test_loader, criterion, device, logger)
    
    # 动态构建测试日志
    test_log_parts = []
    for key in ['loss_E', 'loss_mu', 'loss_m', 'loss_R']:
        if key in test_components:
            test_log_parts.append(f"{key.replace('loss_', '')}: {test_components[key]:.4f}")
    
    if test_log_parts:
        logger.info(
            f"Test Loss: {test_components.get('loss', 0):.4f} "
            f"({', '.join(test_log_parts)}) "
            f"R_sign_acc: {test_components.get('R_sign_acc', 0):.4f}"
        )
    else:
        logger.info(f"Test Loss: {test_components.get('loss', 0):.4f}")

    # Generate test spectra
    logger.info("\n" + "=" * 80)
    logger.info("Generating Test Spectra")
    logger.info("=" * 80)
    spectrum_dir = os.path.join(config['checkpoint_dir'], config['spectrum_dir'])
    generate_test_spectra(model, test_loader, device, spectrum_dir, logger, 
                          num_samples=config['num_test_spectra'])

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    
    # 获取最佳检查点信息 (基于综合评分)
    if pareto_checkpoints:
        def pareto_score(item):
            loss_dict = item[0]
            loss_sum = (loss_dict.get('raw_loss_E', 0) + 
                       loss_dict.get('raw_loss_mu', 0) + 
                       loss_dict.get('raw_loss_m', 0))
            sign_penalty = 1.0 - loss_dict.get('R_sign_acc', 0)
            return loss_sum + sign_penalty * 10
        
        best_pareto = min(pareto_checkpoints, key=pareto_score)
        logger.info(f"Best Pareto checkpoint: Epoch {best_pareto[1]['epoch']} "
                   f"(val_loss: {best_pareto[1]['total_normalized_loss']:.4f})")
        logger.info(f"  Metrics: E={best_pareto[0]['raw_loss_E']:.4f}, "
                   f"mu={best_pareto[0]['raw_loss_mu']:.4f}, "
                   f"m={best_pareto[0]['raw_loss_m']:.4f}, "
                   f"R_sign_acc={best_pareto[0]['R_sign_acc']:.4f}")
    
    logger.info(f"Final test loss: {test_components.get('loss', 0):.4f}")
    logger.info(f"Pareto checkpoints saved to: {config['checkpoint_dir']}")
    logger.info(f"Test spectra saved to: {spectrum_dir}")
    logger.info("=" * 80)

    # Final plot
    plot_path = os.path.join(config['checkpoint_dir'], 'loss_curves_final.png')
    plot_loss_curves(train_losses_history, val_losses_history, plot_path)


if __name__ == '__main__':
    main()
