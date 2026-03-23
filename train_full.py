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
    计算原始损失（未归一化的 MSE）。
    
    Returns:
        dict: 包含原始损失的字典
    """
    import torch.nn.functional as F
    
    batch_size = pred['E_pred'].shape[0]
    
    # 1. 激发能 Loss
    y_E = target['y_E'].reshape(batch_size, n_states)
    loss_E_raw = F.mse_loss(pred['E_pred'], y_E)
    
    # 2. 电跃迁偶极 Loss - Phase-Invariant
    y_mu = target['y_mu_vel'].reshape(batch_size, n_states, 3)
    mu_diff = pred['mu_total'] - y_mu
    mu_sum = pred['mu_total'] + y_mu
    loss_mu_phase1 = (mu_diff ** 2).sum(dim=-1)
    loss_mu_phase2 = (mu_sum ** 2).sum(dim=-1)
    loss_mu_raw = torch.min(loss_mu_phase1, loss_mu_phase2).mean()
    
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
        'loss_mu_raw': loss_mu_raw.item(),
        'loss_m_raw': loss_m_raw.item(),
        'loss_R_raw': loss_R_raw.item(),
    }


def train_epoch(model, loader, criterion, optimizer, device, logger):
    """训练一个 epoch。"""
    model.train()

    total_loss = 0.0
    total_raw_losses = {'loss_E_raw': 0.0, 'loss_mu_raw': 0.0, 'loss_m_raw': 0.0, 'loss_R_raw': 0.0}
    loss_components = {'loss_E': 0.0, 'loss_mu': 0.0, 'loss_m': 0.0, 'loss_R': 0.0, 'loss_R_sign': 0.0}
    total_r_sign_acc = 0.0
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
        
        # Compute raw losses
        raw_losses = compute_raw_losses(pred, target, n_states=model.n_states)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        total_loss += loss_dict['loss']
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        for key in total_raw_losses:
            total_raw_losses[key] += raw_losses[key]
        total_r_sign_acc += loss_dict['R_sign_acc']
        num_batches += 1

        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {total_loss / num_batches:.4f} "
                f"(E: {loss_dict['loss_E']:.4f}, "
                f"mu: {loss_dict['loss_mu']:.4f}, "
                f"m: {loss_dict['loss_m']:.4f}, "
                f"R: {loss_dict['loss_R']:.4f}) "
                f"R_sign_acc: {loss_dict['R_sign_acc']:.4f}"
            )

    # Compute epoch averages
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
    """验证模型。"""
    model.eval()

    total_loss = 0.0
    total_raw_losses = {'loss_E_raw': 0.0, 'loss_mu_raw': 0.0, 'loss_m_raw': 0.0, 'loss_R_raw': 0.0}
    loss_components = {'loss_E': 0.0, 'loss_mu': 0.0, 'loss_m': 0.0, 'loss_R': 0.0, 'loss_R_sign': 0.0}
    total_r_sign_acc = 0.0
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
        
        # Compute raw losses
        raw_losses = compute_raw_losses(pred, target, n_states=model.n_states)

        # Accumulate losses
        total_loss += loss_dict['loss']
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        for key in total_raw_losses:
            total_raw_losses[key] += raw_losses[key]
        total_r_sign_acc += loss_dict['R_sign_acc']
        num_batches += 1

    # Compute averages
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


@torch.no_grad()
def test(model, loader, criterion, device, logger):
    """测试模型。"""
    logger.info("\nRunning test evaluation...")
    return validate(model, loader, criterion, device, logger)


def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot and save training/validation loss curves."""
    fig = plt.figure(figsize=(20, 12))
    
    # Main plot: total loss
    plt.subplot(3, 3, 1)
    epochs = range(1, len(train_losses['total']) + 1)
    plt.plot(epochs, train_losses['total'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['total'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('Training and Validation: Total Normalized Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot: Energy loss (normalized)
    plt.subplot(3, 3, 2)
    plt.plot(epochs, train_losses['E'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['E'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('Excitation Energy Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot: Electric dipole loss (normalized)
    plt.subplot(3, 3, 3)
    plt.plot(epochs, train_losses['mu'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['mu'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('Electric Dipole Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot: Magnetic dipole loss (normalized)
    plt.subplot(3, 3, 4)
    plt.plot(epochs, train_losses['m'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['m'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('Magnetic Dipole Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot: Rotatory strength loss (normalized)
    plt.subplot(3, 3, 5)
    plt.plot(epochs, train_losses['R'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['R'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('Rotatory Strength Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Raw losses
    plt.subplot(3, 3, 6)
    plt.plot(epochs, train_losses['E_raw'], 'b-', label='Train (Raw)', linewidth=2)
    plt.plot(epochs, val_losses['E_raw'], 'r-', label='Val (Raw)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Excitation Energy Loss (Raw MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(3, 3, 7)
    plt.plot(epochs, train_losses['mu_raw'], 'b-', label='Train (Raw)', linewidth=2)
    plt.plot(epochs, val_losses['mu_raw'], 'r-', label='Val (Raw)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Electric Dipole Loss (Raw MSE)')
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
        'checkpoint_dir': 'checkpoints',
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
        'lambda_m': 0.0,
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
        T_max=config['num_epochs']/5
    )

    # Training tracking
    best_val_loss = float('inf')
    train_losses = {'total': [], 'E': [], 'mu': [], 'm': [], 'R': [], 'R_sign_acc': [],
                    'E_raw': [], 'mu_raw': [], 'm_raw': [], 'R_raw': []}
    val_losses = {'total': [], 'E': [], 'mu': [], 'm': [], 'R': [], 'R_sign_acc': [],
                  'E_raw': [], 'mu_raw': [], 'm_raw': [], 'R_raw': []}

    # Training loop
    logger.info("\nStarting training...")
    logger.info("=" * 80)

    # for epoch in range(1, config['num_epochs'] + 1):
    #     logger.info(f"\nEpoch {epoch}/{config['num_epochs']}")
    #     logger.info("-" * 80)

    #     # Train
    #     train_components = train_epoch(
    #         model, train_loader, criterion, optimizer, device, logger
    #     )

    #     # Validate
    #     logger.info("  Running validation...")
    #     val_components = validate(
    #         model, val_loader, criterion, device, logger
    #     )
        
    #     # Update learning rate
    #     scheduler.step()
    #     current_lr = scheduler.get_last_lr()[0]

    #     # Log epoch summary
    #     logger.info("-" * 80)
    #     logger.info(
    #         f"Epoch {epoch} Summary - "
    #         f"Train Loss: {train_components['loss']:.4f}, "
    #         f"Val Loss: {val_components['loss']:.4f}, "
    #         f"LR: {current_lr:.6f}"
    #     )
    #     logger.info(
    #         f"  Raw MSE - Train: E={train_components['loss_E_raw']:.4f}, "
    #         f"mu={train_components['loss_mu_raw']:.4f}, "
    #         f"m={train_components['loss_m_raw']:.4f}, "
    #         f"R={train_components['loss_R_raw']:.4f}"
    #     )
    #     logger.info(
    #         f"  Raw MSE - Val:   E={val_components['loss_E_raw']:.4f}, "
    #         f"mu={val_components['loss_mu_raw']:.4f}, "
    #         f"m={val_components['loss_m_raw']:.4f}, "
    #         f"R={val_components['loss_R_raw']:.4f}"
    #     )
    #     logger.info(
    #         f"  R Sign Acc: Train={train_components['R_sign_acc']:.4f}, "
    #         f"Val={val_components['R_sign_acc']:.4f}"
    #     )

    #     # Store losses for plotting
    #     for key in train_losses:
    #         if key in train_components:
    #             train_losses[key].append(train_components[key])
    #     for key in val_losses:
    #         if key in val_components:
    #             val_losses[key].append(val_components[key])

    #     # Save best model
    #     if val_components['loss'] < best_val_loss:
    #         best_val_loss = val_components['loss']
    #         checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'scheduler_state_dict': scheduler.state_dict(),
    #             'val_loss': val_components['loss'],
    #             'config': config
    #         }, checkpoint_path)
    #         logger.info(f"  Saved best model (val_loss: {val_components['loss']:.4f})")

    #     # Save checkpoint every 10 epochs
    #     if epoch % 100 == 0:
    #         checkpoint_path = os.path.join(
    #             config['checkpoint_dir'],
    #             f'checkpoint_epoch_{epoch}.pt'
    #         )
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'scheduler_state_dict': scheduler.state_dict(),
    #             'val_loss': val_components['loss'],
    #             'config': config
    #         }, checkpoint_path)
    #         logger.info(f"  Saved checkpoint at epoch {epoch}")

    #     # Plot loss curves
    #     if epoch % 5 == 0 or epoch == 1:
    #         plot_path = os.path.join(config['checkpoint_dir'], 'loss_curves.png')
    #         plot_loss_curves(train_losses, val_losses, plot_path)
    #         logger.info(f"  Updated loss curves plot")
    
    # 加载训练好的模型进行测试
    last_checkpoint = torch.load('checkpoints/checkpoint_epoch_1000.pt')
    model.load_state_dict(last_checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {last_checkpoint['epoch']}")
    
    # Final test evaluation
    logger.info("\n" + "=" * 80)
    logger.info("Final Test Evaluation")
    logger.info("=" * 80)
    test_components = test(model, test_loader, criterion, device, logger)
    logger.info(
        f"Test Loss: {test_components['loss']:.4f} "
        f"(E: {test_components['loss_E']:.4f}, "
        f"mu: {test_components['loss_mu']:.4f}, "
        f"m: {test_components['loss_m']:.4f}, "
        f"R: {test_components['loss_R']:.4f}) "
        f"R_sign_acc: {test_components['R_sign_acc']:.4f}"
    )

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
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final test loss: {test_components['loss']:.4f}")
    logger.info(f"Best model saved to: {os.path.join(config['checkpoint_dir'], 'best_model.pt')}")
    logger.info(f"Test spectra saved to: {spectrum_dir}")
    logger.info("=" * 80)

    # Final plot
    plot_path = os.path.join(config['checkpoint_dir'], 'loss_curves_final.png')
    plot_loss_curves(train_losses, val_losses, plot_path)


if __name__ == '__main__':
    main()
