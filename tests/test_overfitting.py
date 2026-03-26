"""
过拟合测试脚本 - 测试模型在单个 batch 上的拟合能力。

完全照搬 scripts/03_train.py 的超参数设置，
只使用训练集的一个 batch 进行训练和验证。
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
    log_file = os.path.join(log_dir, 'overfit_test.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_single_batch(data_dir, batch_size=64, num_workers=4):
    """从训练集中加载一个 batch 用于过拟合测试。"""
    logger = logging.getLogger(__name__)

    train_path = os.path.join(data_dir, 'train.pt')

    logger.info(f"Loading training data from {train_path}")
    train_data = torch.load(train_path, weights_only=False)
    logger.info(f"Loaded {len(train_data)} training samples")

    # 创建 dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,  # 不打乱，取第一个 batch
        num_workers=num_workers,
        pin_memory=True
    )

    # 只取第一个 batch
    single_batch = next(iter(train_loader))
    logger.info(f"Selected single batch with {single_batch.num_graphs} molecules")

    # 验证集和训练集使用同一个 batch
    return single_batch, single_batch


def compute_raw_losses(pred, target, n_states=20):
    """
    计算原始损失（未归一化的 MSE）。
    
    Returns:
        dict: 包含原始损失的字典
    """
    import torch.nn.functional as F
    
    batch_size = pred['E_pred'].shape[0]
    
    # 1. 激发能 Loss
    y_E = target['y_E'].reshape(batch_size, n_states)
    loss_E_raw = F.mse_loss(pred['E_pred'], y_E)
    
    # 2. 速度电跃迁偶极 Loss - Phase-Invariant
    y_mu_vel = target['y_mu_vel'].reshape(batch_size, n_states, 3)
    mu_vel_diff = pred['mu_total_vel'] - y_mu_vel
    mu_vel_sum = pred['mu_total_vel'] + y_mu_vel
    loss_mu_vel_phase1 = (mu_vel_diff ** 2).sum(dim=-1)
    loss_mu_vel_phase2 = (mu_vel_sum ** 2).sum(dim=-1)
    loss_mu_vel_raw = torch.min(loss_mu_vel_phase1, loss_mu_vel_phase2).mean()
    
    # 3. 磁跃迁偶极 Loss - Phase-Invariant
    y_m = target['y_m'].reshape(batch_size, n_states, 3)
    m_diff = pred['m_total'] - y_m
    m_sum = pred['m_total'] + y_m
    loss_m_phase1 = (m_diff ** 2).sum(dim=-1)
    loss_m_phase2 = (m_sum ** 2).sum(dim=-1)
    loss_m_raw = torch.min(loss_m_phase1, loss_m_phase2).mean()
    
    # 4. 旋转强度 R Loss
    y_R = target['y_R'].reshape(batch_size, n_states)
    loss_R_raw = F.mse_loss(pred['R_pred'], y_R)
    
    return {
        'loss_E_raw': loss_E_raw.item(),
        'loss_mu_vel_raw': loss_mu_vel_raw.item(),
        'loss_m_raw': loss_m_raw.item(),
        'loss_R_raw': loss_R_raw.item(),
    }


def train_step(model, data, criterion, optimizer, device, logger):
    """单步训练。"""
    model.train()

    # Move data to device
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
    
    # Compute raw losses (unweighted)
    raw_losses = compute_raw_losses(pred, target, n_states=model.n_states)
    loss_dict.update(raw_losses)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping to prevent NaN
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return loss_dict


@torch.no_grad()
def validate(model, data, criterion, device, logger):
    """验证模型（在相同的 batch 上）。"""
    model.eval()

    # Move data to device
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
    
    # Compute raw losses (unweighted)
    raw_losses = compute_raw_losses(pred, target, n_states=model.n_states)
    loss_dict.update(raw_losses)

    return loss_dict


def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot and save training/validation loss curves."""
    fig = plt.figure(figsize=(20, 12))
    
    # Main plot: total loss
    plt.subplot(3, 3, 1)
    steps = range(1, len(train_losses['total']) + 1)
    plt.plot(steps, train_losses['total'], 'b-', label='Train', linewidth=2)
    plt.plot(steps, val_losses['total'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Normalized Loss')
    plt.title('Overfitting Test: Total Normalized Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot: Energy loss (normalized)
    plt.subplot(3, 3, 2)
    plt.plot(steps, train_losses['E'], 'b-', label='Train', linewidth=2)
    plt.plot(steps, val_losses['E'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Normalized Loss')
    plt.title('Excitation Energy Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot: Velocity electric dipole loss (normalized)
    plt.subplot(3, 3, 3)
    plt.plot(steps, train_losses['mu_vel'], 'b-', label='Train', linewidth=2)
    plt.plot(steps, val_losses['mu_vel'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Normalized Loss')
    plt.title('Velocity Electric Dipole Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot: Magnetic dipole loss (normalized)
    plt.subplot(3, 3, 4)
    plt.plot(steps, train_losses['m'], 'b-', label='Train', linewidth=2)
    plt.plot(steps, val_losses['m'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Normalized Loss')
    plt.title('Magnetic Dipole Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot: Rotatory strength loss (normalized)
    plt.subplot(3, 3, 5)
    plt.plot(steps, train_losses['R'], 'b-', label='Train', linewidth=2)
    plt.plot(steps, val_losses['R'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Normalized Loss')
    plt.title('Rotatory Strength Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Raw losses
    plt.subplot(3, 3, 6)
    plt.plot(steps, train_losses['E_raw'], 'b-', label='Train (Raw)', linewidth=2)
    plt.plot(steps, val_losses['E_raw'], 'r-', label='Val (Raw)', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.title('Excitation Energy Loss (Raw MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(3, 3, 7)
    plt.plot(steps, train_losses['mu_vel_raw'], 'b-', label='Train (Raw)', linewidth=2)
    plt.plot(steps, val_losses['mu_vel_raw'], 'r-', label='Val (Raw)', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.title('Velocity Electric Dipole Loss (Raw MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(3, 3, 8)
    plt.plot(steps, train_losses['m_raw'], 'b-', label='Train (Raw)', linewidth=2)
    plt.plot(steps, val_losses['m_raw'], 'r-', label='Val (Raw)', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.title('Magnetic Dipole Loss (Raw MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(3, 3, 9)
    plt.plot(steps, train_losses['R_raw'], 'b-', label='Train (Raw)', linewidth=2)
    plt.plot(steps, val_losses['R_raw'], 'r-', label='Val (Raw)', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.title('Rotatory Strength Loss (Raw MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_spectrum(E_pred, R_pred_au, wavelength_grid, sigma=0.4):
    """
    应用高斯展宽生成连续 ECD 光谱。
    
    物理公式:
    1. R_cgs = R_au × 471.44 (in 10^-40 cgs units)
    2. E_grid = 1240 / λ (eV)
    3. Δε(E) = (1 / (2.296×10^1 × σ × √π)) × Σ E_i × R_cgs,i × exp[-(E - E_i)^2 / σ^2]
    4. [θ] = Δε × 3298.2
    
    Args:
        E_pred: [20] 激发能量 (eV)
        R_pred_au: [20] 旋光强度 (原子单位)
        wavelength_grid: [N] 波长值 (nm)
        sigma: 高斯展宽宽度 (eV)
    
    Returns:
        molar_ellipticity: [N] 摩尔椭圆度值 (deg·cm^2/dmol)
    """
    # Step 1: 将 R 从原子单位转换为 cgs 单位
    R_cgs = R_pred_au * 471.44  # 结果是 10^-40 cgs 单位
    
    # Step 2: 将波长网格转换为能量网格
    E_grid = 1240.0 / wavelength_grid  # [N] 能量 (eV)
    
    # Step 3: 高斯展宽计算 Δε
    norm_constant = 2.296e1 * sigma * np.sqrt(np.pi)
    delta_epsilon = np.zeros_like(E_grid)
    
    # 对所有 20 个激发态求和
    for i in range(len(E_pred)):
        gaussian = np.exp(-((E_grid - E_pred[i]) / sigma) ** 2)
        delta_epsilon += E_pred[i] * R_cgs[i] * gaussian
    
    # 应用归一化
    delta_epsilon /= norm_constant
    
    # Step 4: 将 Δε 转换为 [θ] (摩尔椭圆度)
    molar_ellipticity = delta_epsilon * 3298.2
    
    return molar_ellipticity


def plot_spectrum_comparison(wavelength, pred_spectrum, real_spectrum, mol_id, 
                              save_path, title=None):
    """
    绘制预测光谱和真实光谱的对比图。
    
    Args:
        wavelength: [N] 波长值 (nm)
        pred_spectrum: [N] 预测光谱值
        real_spectrum: [N] 真实光谱值
        mol_id: 分子 ID
        save_path: 保存路径
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 绘制真实光谱
    ax.plot(wavelength, real_spectrum, 'b-', linewidth=2.0, label='Real', alpha=0.8)
    
    # 绘制预测光谱
    ax.plot(wavelength, pred_spectrum, 'r--', linewidth=2.0, label='Predicted', alpha=0.8)
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # 设置标签
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('[θ] (deg·cm²/dmol)', fontsize=12, fontweight='bold')
    
    # 设置标题
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title(f'Molecule {mol_id}: ECD Spectrum Comparison', fontsize=14, fontweight='bold', pad=15)
    
    # 设置 x 轴范围
    ax.set_xlim(wavelength.min(), wavelength.max())
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加图例
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # 美化刻度
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_and_plot_spectra(model, data, device, output_dir, logger, num_samples=5):
    """
    为前 num_samples 个分子生成并绘制光谱对比图。
    """
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建波长网格
    wavelength_min, wavelength_max, wavelength_step = 80.0, 450.0, 1.0
    wavelength_grid = np.arange(wavelength_min, wavelength_max + wavelength_step, wavelength_step)
    sigma = 0.4
    
    with torch.no_grad():
        data = data.to(device)
        pred = model(data)
        
        batch_size = data.num_graphs
        n_states = model.n_states
        
        logger.info(f"\nGenerating spectra for first {min(num_samples, batch_size)} molecules...")
        
        for i in range(min(num_samples, batch_size)):
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
            
            # 将真实 R 从 cgs 转换为原子单位用于一致性
            R_real_au = R_real_cgs / 471.44
            R_pred_au = R_pred_au / 471.44
            
            # 生成预测光谱
            pred_spectrum = generate_spectrum(E_pred, R_pred_au, wavelength_grid, sigma)
            
            # 生成真实光谱
            real_spectrum = generate_spectrum(E_real, R_real_au, wavelength_grid, sigma)
            
            # 获取分子 ID
            mol_id = data.mol_id[i].item() if hasattr(data, 'mol_id') else i
            
            # 绘制对比图
            save_path = os.path.join(output_dir, f'spectrum_comparison_mol_{mol_id}.png')
            plot_spectrum_comparison(wavelength_grid, pred_spectrum, real_spectrum, 
                                    mol_id, save_path)
            
            logger.info(f"  Saved spectrum comparison for molecule {mol_id}")
            
            # 计算并记录光谱差异
            spectrum_diff = np.mean(np.abs(pred_spectrum - real_spectrum))
            logger.info(f"    Mean absolute difference: {spectrum_diff:.4e}")


def print_predictions_vs_targets(model, data, device, logger, num_samples=3):
    """打印预测值与真实值的对比。"""
    model.eval()
    
    with torch.no_grad():
        data = data.to(device)
        pred = model(data)
        
        batch_size = data.num_graphs
        n_states = model.n_states
        
        for i in range(min(num_samples, batch_size)):
            logger.info(f"\n--- Sample {i+1} ---")
            
            E_pred = pred['E_pred'][i].cpu().numpy()
            
            if data.y_E.dim() == 2:
                E_target = data.y_E[i].cpu().numpy()
                R_target = data.y_R[i].cpu().numpy()
            else:
                start_idx = i * n_states
                end_idx = start_idx + n_states
                E_target = data.y_E[start_idx:end_idx].cpu().numpy()
                R_target = data.y_R[start_idx:end_idx].cpu().numpy()
            
            R_pred = pred['R_pred'][i].cpu().numpy()
            
            logger.info(f"Excitation Energies (eV):")
            for j in range(min(5, len(E_pred))):
                logger.info(f"  State {j+1}: Pred={E_pred[j]:.4f}, Target={E_target[j]:.4f}, Diff={abs(E_pred[j]-E_target[j]):.4f}")
            
            logger.info(f"Rotatory Strengths (10^-40 cgs):")
            for j in range(min(5, len(R_pred))):
                logger.info(f"  State {j+1}: Pred={R_pred[j]:+.4f}, Target={R_target[j]:+.4f}, Diff={abs(R_pred[j]-R_target[j]):.4f}")


def main():
    """过拟合测试主函数。"""
    # Configuration - 完全照搬 03_train.py
    config = {
        'data_dir': 'data/processed_with_enantiomers',
        'output_dir': 'overfit_test_results',
        'batch_size': 64,
        'num_steps': 5000,  # 训练步数（替代 epoch）
        'eval_interval': 50,  # 每隔多少步验证一次
        'learning_rate': 2e-3,
        'weight_decay': 1e-5,
        'num_workers': 8,
        # Model hyperparameters
        'num_features': 128,
        'max_l': 2,
        'num_blocks': 3,
        'num_radial': 32,
        'cutoff': 50.0,
        'n_states': 20,
        'max_atomic_number': 60,
        
        # Loss Weights（归一化后的相对权重）
        'lambda_E': 1.0,
        'lambda_mu_vel': 1.0,   # 联合 μ-m 相位损失权重
        'lambda_m': 1.0,        # 保留参数，联合约束中已隐含
        'lambda_R': 1.0,
        'lambda_R_sign': 0.0,
    }

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(config['output_dir'])

    logger.info("=" * 80)
    logger.info("PhysECD Overfitting Test")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Configuration: {config}")
    logger.info("\n注意：本测试使用训练集的单个 batch 进行训练和验证")
    logger.info("如果模型拟合能力足够，训练损失应该趋近于 0")

    # Load single batch
    logger.info("\nLoading single batch from training set...")
    train_batch, val_batch = load_single_batch(
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
        lambda_mu_vel=config['lambda_mu_vel'],
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
        T_max=config['num_steps'],
        eta_min=1e-4
    )

    # Training tracking (包含原始损失和归一化损失)
    train_losses = {
        'total': [], 'E': [], 'mu_vel': [], 'm': [], 'R': [], 'R_sign_acc': [],
        'E_raw': [], 'mu_vel_raw': [], 'm_raw': [], 'R_raw': []
    }
    val_losses = {
        'total': [], 'E': [], 'mu_vel': [], 'm': [], 'R': [], 'R_sign_acc': [],
        'E_raw': [], 'mu_vel_raw': [], 'm_raw': [], 'R_raw': []
    }

    # Training loop
    logger.info("\nStarting overfitting test...")
    logger.info("=" * 80)

    start_time = time.time()

    for step in range(1, config['num_steps'] + 1):
        # Train
        train_components = train_step(model, train_batch, criterion, optimizer, device, logger)
        
        # Update learning rate
        scheduler.step()

        # Evaluate
        if step % config['eval_interval'] == 0 or step == 1:
            val_components = validate(model, val_batch, criterion, device, logger)
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # Store losses (归一化损失)
            train_losses['total'].append(train_components['loss'])
            train_losses['E'].append(train_components['loss_E'])
            train_losses['mu_vel'].append(train_components['loss_mu_vel'])
            train_losses['m'].append(train_components['loss_m'])
            train_losses['R'].append(train_components['loss_R'])
            train_losses['R_sign_acc'].append(train_components['R_sign_acc'])
            
            # Store raw losses (原始 MSE)
            train_losses['E_raw'].append(train_components['loss_E_raw'])
            train_losses['mu_vel_raw'].append(train_components['loss_mu_vel_raw'])
            train_losses['m_raw'].append(train_components['loss_m_raw'])
            train_losses['R_raw'].append(train_components['loss_R_raw'])

            val_losses['total'].append(val_components['loss'])
            val_losses['E'].append(val_components['loss_E'])
            val_losses['mu_vel'].append(val_components['loss_mu_vel'])
            val_losses['m'].append(val_components['loss_m'])
            val_losses['R'].append(val_components['loss_R'])
            val_losses['R_sign_acc'].append(val_components['R_sign_acc'])
            
            # Store raw losses (原始 MSE)
            val_losses['E_raw'].append(val_components['loss_E_raw'])
            val_losses['mu_vel_raw'].append(val_components['loss_mu_vel_raw'])
            val_losses['m_raw'].append(val_components['loss_m_raw'])
            val_losses['R_raw'].append(val_components['loss_R_raw'])

            elapsed = time.time() - start_time

            logger.info(f"\nStep {step}/{config['num_steps']} ({elapsed:.1f}s, LR: {current_lr:.6f})")
            logger.info(f"  Normalized Loss - Train: {train_components['loss']:.6f}, Val: {val_components['loss']:.6f}")
            logger.info(f"  Raw MSE Loss    - E: {train_components['loss_E_raw']:.6f}/{val_components['loss_E_raw']:.6f}, "
                       f"mu_vel: {train_components['loss_mu_vel_raw']:.6f}/{val_components['loss_mu_vel_raw']:.6f}, "
                       f"m: {train_components['loss_m_raw']:.6f}/{val_components['loss_m_raw']:.6f}, "
                       f"R: {train_components['loss_R_raw']:.6f}/{val_components['loss_R_raw']:.6f}")
            logger.info(f"  R Sign Acc: Train={train_components['R_sign_acc']:.4f}, Val={val_components['R_sign_acc']:.4f}")

            # 更新损失曲线图
            plot_path = os.path.join(config['output_dir'], 'loss_curves.png')
            plot_loss_curves(train_losses, val_losses, plot_path)

    # Final evaluation
    logger.info("\n" + "=" * 80)
    logger.info("Overfitting test completed!")
    logger.info("=" * 80)

    # 打印最终的预测与目标对比
    logger.info("\nFinal Predictions vs Targets (first 3 samples):")
    print_predictions_vs_targets(model, train_batch, device, logger, num_samples=3)

    # 生成并绘制前5个分子的光谱对比图
    logger.info("\n" + "=" * 80)
    logger.info("Generating spectrum comparisons...")
    logger.info("=" * 80)
    generate_and_plot_spectra(model, train_batch, device, config['output_dir'], logger, num_samples=5)

    # Final plot
    plot_path = os.path.join(config['output_dir'], 'loss_curves_final.png')
    plot_loss_curves(train_losses, val_losses, plot_path)
    logger.info(f"\nLoss curves saved to: {plot_path}")

    # 保存模型
    model_path = os.path.join(config['output_dir'], 'overfit_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'final_train_loss': train_losses['total'][-1],
        'final_val_loss': val_losses['total'][-1],
    }, model_path)
    logger.info(f"Model saved to: {model_path}")

    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("Summary:")
    logger.info("=" * 80)
    logger.info(f"Initial normalized loss: {train_losses['total'][0]:.6f}")
    logger.info(f"Final normalized loss:   {train_losses['total'][-1]:.6f}")
    logger.info(f"Initial raw E loss:      {train_losses['E_raw'][0]:.6f}")
    logger.info(f"Final raw E loss:        {train_losses['E_raw'][-1]:.6f}")
    logger.info(f"Initial raw R loss:      {train_losses['R_raw'][0]:.6f}")
    logger.info(f"Final raw R loss:        {train_losses['R_raw'][-1]:.6f}")


if __name__ == '__main__':
    main()
