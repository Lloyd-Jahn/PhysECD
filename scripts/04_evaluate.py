"""
评估脚本（支持 train / val / test 三个分割）

加载训练好的模型 checkpoint，在指定数据分割上计算各项指标，
并批量生成预测光谱 vs 真实光谱对比图。

运行指令（从项目根目录）：
  # 评估测试集（默认）
  python scripts/04_evaluate.py

  # 评估训练集（用于确认过拟合程度）
  python scripts/04_evaluate.py --split train

  # 评估验证集，指定 checkpoint
  python scripts/04_evaluate.py --split val --checkpoint checkpoints/checkpoint_epoch_1000.pt
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from torch_geometric.loader import DataLoader

# 将项目根目录加入 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physecd.models import PhysECDModel
from physecd.physics import PhysECDLoss


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='Evaluate PhysECD model on train / val / test split'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='要评估的数据分割：train / val / test（默认：test）'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='模型 checkpoint 文件路径（默认：checkpoints/best_model.pt）'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed_with_enantiomers',
        help='数据集目录（默认：data/processed_with_enantiomers）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认：checkpoints/{split}_evaluation/）'
    )
    parser.add_argument(
        '--num_spectra',
        type=int,
        default=100,
        help='生成光谱对比图的分子数量（默认：100）'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='推断时的 batch size（默认：64）'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader 的 worker 数量（默认：4）'
    )
    return parser.parse_args()


def setup_logging(output_dir):
    """初始化日志，同时输出到文件和控制台。"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'evaluate.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def build_model_and_criterion(config, device):
    """根据 checkpoint 中保存的 config 重建模型和损失函数。"""
    model = PhysECDModel(
        num_features=config['num_features'],
        max_l=config['max_l'],
        num_blocks=config['num_blocks'],
        num_radial=config['num_radial'],
        cutoff=config['cutoff'],
        n_states=config['n_states'],
        max_atomic_number=config['max_atomic_number']
    ).to(device)

    criterion = PhysECDLoss(
        lambda_E=config['lambda_E'],
        lambda_mu_vel=config['lambda_mu_vel'],
        lambda_m=config['lambda_m'],
        lambda_R=config['lambda_R'],
        lambda_R_sign=config.get('lambda_R_sign', 0.0)
    ).to(device)

    return model, criterion


def compute_raw_losses(pred, target, n_states):
    """计算原始损失（未归一化的 MSE），用于指标报告。"""
    batch_size = pred['E_pred'].shape[0]

    y_E = target['y_E'].reshape(batch_size, n_states)
    loss_E_raw = F.mse_loss(pred['E_pred'], y_E)

    y_mu_vel = target['y_mu_vel'].reshape(batch_size, n_states, 3)
    mu_vel_diff = pred['mu_total_vel'] - y_mu_vel
    mu_vel_sum  = pred['mu_total_vel'] + y_mu_vel
    loss_mu_vel_raw = torch.min(
        (mu_vel_diff ** 2).sum(dim=-1),
        (mu_vel_sum  ** 2).sum(dim=-1)
    ).mean()

    y_m = target['y_m'].reshape(batch_size, n_states, 3)
    m_diff = pred['m_total'] - y_m
    m_sum  = pred['m_total'] + y_m
    loss_m_raw = torch.min(
        (m_diff ** 2).sum(dim=-1),
        (m_sum  ** 2).sum(dim=-1)
    ).mean()

    y_R = target['y_R'].reshape(batch_size, n_states)
    loss_R_raw = F.mse_loss(pred['R_pred'], y_R)

    return {
        'loss_E_raw':      loss_E_raw.item(),
        'loss_mu_vel_raw': loss_mu_vel_raw.item(),
        'loss_m_raw':      loss_m_raw.item(),
        'loss_R_raw':      loss_R_raw.item(),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, logger):
    """在数据集上运行完整评估，返回归一化损失和原始损失。"""
    model.eval()

    total_loss = 0.0
    total_raw = {'loss_E_raw': 0.0, 'loss_mu_vel_raw': 0.0,
                 'loss_m_raw': 0.0, 'loss_R_raw': 0.0}
    components = {'loss_E': 0.0, 'loss_mu_vel': 0.0,
                  'loss_m': 0.0, 'loss_R': 0.0, 'loss_R_sign': 0.0}
    total_r_sign_acc = 0.0
    num_batches = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = {
            'y_E':      data.y_E,
            'y_mu_vel': data.y_mu_vel,
            'y_m':      data.y_m,
            'y_R':      data.y_R
        }
        loss, loss_dict = criterion(pred, target)
        raw = compute_raw_losses(pred, target, model.n_states)

        total_loss += loss_dict['loss']
        for k in components:
            components[k] += loss_dict[k]
        for k in total_raw:
            total_raw[k] += raw[k]
        total_r_sign_acc += loss_dict['R_sign_acc']
        num_batches += 1

    components['loss']       = total_loss / num_batches
    components['R_sign_acc'] = total_r_sign_acc / num_batches
    for k in list(components.keys()):
        if k not in ('loss', 'R_sign_acc'):
            components[k] /= num_batches
    for k in total_raw:
        total_raw[k] /= num_batches
    components.update(total_raw)

    return components


def generate_spectrum(E, R_cgs, wavelength_grid, sigma=0.4):
    """
    高斯展宽：将离散激发态 (E, R) 展宽为连续 ECD 光谱。

    物理公式：
      E_grid = 1240 / λ  (eV)
      Δε(E) = 1/(2.296×10¹ × σ × √π) × Σ_i E_i × R_cgs,i × exp[-(E-E_i)²/σ²]
      [θ] = Δε × 3298.2

    Args:
        E:              [n_states] 激发能（eV）
        R_cgs:          [n_states] 旋转强度（10^-40 cgs）
        wavelength_grid:[N] 波长网格（nm）
        sigma:          高斯展宽宽度（eV，默认 0.4）

    Returns:
        [N] 摩尔椭圆度（deg·cm²/dmol）
    """
    E_grid = 1240.0 / wavelength_grid
    norm = 2.296e1 * sigma * np.sqrt(np.pi)
    delta_eps = np.zeros_like(E_grid)
    for i in range(len(E)):
        delta_eps += E[i] * R_cgs[i] * np.exp(-((E_grid - E[i]) / sigma) ** 2)
    return delta_eps / norm * 3298.2


def plot_spectrum_comparison(wavelength, pred_spectrum, real_spectrum,
                             mol_id, save_path, title=None):
    """绘制预测光谱与真实光谱的对比图。"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(wavelength, real_spectrum, 'b-',  linewidth=2.0, label='Real',      alpha=0.8)
    ax.plot(wavelength, pred_spectrum, 'r--', linewidth=2.0, label='Predicted', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('[θ] (deg·cm²/dmol)', fontsize=12, fontweight='bold')
    ax.set_title(
        title if title else f'Molecule {mol_id}: ECD Spectrum Comparison',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.set_xlim(wavelength.min(), wavelength.max())
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def generate_test_spectra(model, loader, device, output_dir, logger, num_samples=100):
    """
    为数据集前 num_samples 个分子生成预测 vs 真实光谱对比图，
    并输出光谱差异统计。
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    wavelength_grid = np.arange(80.0, 451.0, 1.0)  # 80~450 nm，步长 1 nm
    sigma = 0.4

    logger.info(f"Generating spectra for first {num_samples} molecules...")
    logger.info(f"Output directory: {output_dir}")

    count = 0
    errors = []

    for data in loader:
        if count >= num_samples:
            break

        data = data.to(device)
        pred = model(data)
        n_states = model.n_states

        for i in range(data.num_graphs):
            if count >= num_samples:
                break

            E_pred = pred['E_pred'][i].cpu().numpy()
            R_pred = pred['R_pred'][i].cpu().numpy()

            # 真实值（10^-40 cgs）
            if data.y_E.dim() == 2:
                E_real = data.y_E[i].cpu().numpy()
                R_real = data.y_R[i].cpu().numpy()
            else:
                s, e = i * n_states, (i + 1) * n_states
                E_real = data.y_E[s:e].cpu().numpy()
                R_real = data.y_R[s:e].cpu().numpy()

            pred_spectrum = generate_spectrum(E_pred, R_pred, wavelength_grid, sigma)
            real_spectrum = generate_spectrum(E_real, R_real, wavelength_grid, sigma)

            mol_id = data.mol_id[i].item() if hasattr(data, 'mol_id') else count
            save_path = os.path.join(output_dir, f'spectrum_mol_{mol_id}.png')
            plot_spectrum_comparison(wavelength_grid, pred_spectrum, real_spectrum,
                                     mol_id, save_path)

            errors.append(np.mean(np.abs(pred_spectrum - real_spectrum)))
            count += 1

            if count % 10 == 0:
                logger.info(f"  Generated {count}/{num_samples} spectra...")

    # 统计汇总
    mean_err   = np.mean(errors)
    median_err = np.median(errors)
    std_err    = np.std(errors)

    logger.info(f"\nSpectrum generation complete ({count} molecules)")
    logger.info(f"  Mean   |Δ[θ]|: {mean_err:.4e}")
    logger.info(f"  Median |Δ[θ]|: {median_err:.4e}")
    logger.info(f"  Std    |Δ[θ]|: {std_err:.4e}")

    stats_path = os.path.join(output_dir, 'spectrum_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Spectrum Comparison Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total spectra: {count}\n")
        f.write(f"Mean   |Δ[θ]|: {mean_err:.4e}\n")
        f.write(f"Median |Δ[θ]|: {median_err:.4e}\n")
        f.write(f"Std    |Δ[θ]|: {std_err:.4e}\n")
        f.write("\nPer-molecule errors:\n")
        for i, err in enumerate(errors):
            f.write(f"  {i:4d}: {err:.4e}\n")
    logger.info(f"  Statistics saved to: {stats_path}")


@torch.no_grad()
def collect_predictions(model, loader, device):
    """遍历整个数据集，收集所有预测值和真实值（用于定量评估）。"""
    model.eval()
    n_states = model.n_states

    all_mol_ids = []
    all_smiles  = []
    all_E_pred  = []
    all_E_true  = []
    all_R_pred  = []
    all_R_true  = []

    for data in loader:
        data = data.to(device)
        pred = model(data)
        batch_size = pred['E_pred'].shape[0]

        all_mol_ids.extend(data.mol_id.cpu().tolist())
        all_smiles.extend(data.smiles)

        E_pred_np = pred['E_pred'].cpu().numpy()   # [B, 20]
        R_pred_np = pred['R_pred'].cpu().numpy()   # [B, 20]

        if data.y_E.dim() == 2:
            E_true_np = data.y_E.cpu().numpy()
            R_true_np = data.y_R.cpu().numpy()
        else:
            E_true_np = data.y_E.cpu().numpy().reshape(batch_size, n_states)
            R_true_np = data.y_R.cpu().numpy().reshape(batch_size, n_states)

        all_E_pred.append(E_pred_np)
        all_E_true.append(E_true_np)
        all_R_pred.append(R_pred_np)
        all_R_true.append(R_true_np)

    return {
        'mol_ids': all_mol_ids,
        'smiles':  all_smiles,
        'E_pred':  np.concatenate(all_E_pred, axis=0),   # [N, 20]
        'E_true':  np.concatenate(all_E_true, axis=0),
        'R_pred':  np.concatenate(all_R_pred, axis=0),
        'R_true':  np.concatenate(all_R_true, axis=0),
    }


def _plot_pcc_histogram(pcc_values, output_dir, logger):
    """绘制光谱 PCC 分布直方图。"""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(pcc_values, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(pcc_values), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(pcc_values):.3f}')
    ax.axvline(np.median(pcc_values), color='orange', linestyle='--', linewidth=1.5,
               label=f'Median = {np.median(pcc_values):.3f}')
    ax.set_xlabel('Spectrum PCC', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Per-Molecule Spectrum PCC', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pcc_histogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def _plot_scatter(x_true, x_pred, pearson_r, xlabel, ylabel, title, save_path, logger):
    """绘制预测 vs 真实值散点图，含 y=x 参考线和 Pearson 相关系数标注。"""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.scatter(x_true, x_pred, alpha=0.1, s=1, color='steelblue', rasterized=True)
    lo = min(x_true.min(), x_pred.min())
    hi = max(x_true.max(), x_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, alpha=0.8, label='y = x')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def _plot_best_worst_spectra(pcc_values, mol_ids, E_pred, E_true, R_pred, R_true,
                              wavelength_grid, output_dir, logger, n=5):
    """绘制 PCC 最高/最低 n 个分子的光谱对比图。"""
    sorted_idx = np.argsort(pcc_values)
    worst_idx  = sorted_idx[:n]
    best_idx   = sorted_idx[-n:][::-1]

    for rank, idx in enumerate(best_idx, start=1):
        pred_spec = generate_spectrum(E_pred[idx], R_pred[idx], wavelength_grid)
        real_spec = generate_spectrum(E_true[idx], R_true[idx], wavelength_grid)
        save_path = os.path.join(output_dir, f'best_spectrum_{rank}.png')
        plot_spectrum_comparison(
            wavelength_grid, pred_spec, real_spec, mol_ids[idx], save_path,
            title=f'Best #{rank}  mol_id={mol_ids[idx]}  PCC={pcc_values[idx]:.4f}'
        )
    logger.info(f"  Saved top-{n} best spectra")

    for rank, idx in enumerate(worst_idx, start=1):
        pred_spec = generate_spectrum(E_pred[idx], R_pred[idx], wavelength_grid)
        real_spec = generate_spectrum(E_true[idx], R_true[idx], wavelength_grid)
        save_path = os.path.join(output_dir, f'worst_spectrum_{rank}.png')
        plot_spectrum_comparison(
            wavelength_grid, pred_spec, real_spec, mol_ids[idx], save_path,
            title=f'Worst #{rank}  mol_id={mol_ids[idx]}  PCC={pcc_values[idx]:.4f}'
        )
    logger.info(f"  Saved top-{n} worst spectra")


def run_quantitative_evaluation(collected, output_dir, logger):
    """
    根据收集的预测数据计算定量指标，并保存所有输出文件到 output_dir/quantitative/。

    输出文件：
      - per_molecule_metrics.csv  逐分子指标明细
      - quantitative_summary.txt  全局定量汇总
      - pcc_histogram.png         PCC 分布直方图
      - R_scatter.png             R 预测 vs 真实散点图
      - E_scatter.png             E 预测 vs 真实散点图
      - best_spectrum_{1-5}.png   PCC 最高 5 个分子的光谱对比图
      - worst_spectrum_{1-5}.png  PCC 最低 5 个分子的光谱对比图
    """
    quant_dir = os.path.join(output_dir, 'quantitative')
    os.makedirs(quant_dir, exist_ok=True)

    mol_ids = collected['mol_ids']
    smiles  = collected['smiles']
    E_pred  = collected['E_pred']   # [N, 20]
    E_true  = collected['E_true']
    R_pred  = collected['R_pred']
    R_true  = collected['R_true']
    N = len(mol_ids)

    wavelength_grid = np.arange(80.0, 451.0, 1.0)  # 80~450 nm，371 个点

    # --- 1. 全局离散态指标（展平所有 N×20 个数据点）---
    E_flat_pred = E_pred.flatten()
    E_flat_true = E_true.flatten()
    R_flat_pred = R_pred.flatten()
    R_flat_true = R_true.flatten()

    e_mae_global = float(np.mean(np.abs(E_flat_pred - E_flat_true)))
    r_mae_global = float(np.mean(np.abs(R_flat_pred - R_flat_true)))
    r_pearson_global, _ = scipy_stats.pearsonr(R_flat_pred, R_flat_true)
    e_pearson_global, _ = scipy_stats.pearsonr(E_flat_pred, E_flat_true)

    # --- 2. 逐分子光谱 PCC ---
    logger.info(f"  Computing per-molecule spectrum PCC for {N} molecules...")
    per_mol_pcc   = []
    per_mol_r_mae = []
    per_mol_e_mae = []

    for i in range(N):
        pred_spec = generate_spectrum(E_pred[i], R_pred[i], wavelength_grid)
        real_spec = generate_spectrum(E_true[i], R_true[i], wavelength_grid)
        pcc, _ = scipy_stats.pearsonr(pred_spec, real_spec)
        per_mol_pcc.append(pcc)
        per_mol_r_mae.append(float(np.mean(np.abs(R_pred[i] - R_true[i]))))
        per_mol_e_mae.append(float(np.mean(np.abs(E_pred[i] - E_true[i]))))

    per_mol_pcc = np.array(per_mol_pcc)

    pcc_mean   = float(np.mean(per_mol_pcc))
    pcc_median = float(np.median(per_mol_pcc))
    pcc_std    = float(np.std(per_mol_pcc))
    pcc_p10    = float(np.percentile(per_mol_pcc, 10))
    pcc_p25    = float(np.percentile(per_mol_pcc, 25))
    pcc_p75    = float(np.percentile(per_mol_pcc, 75))
    pcc_p90    = float(np.percentile(per_mol_pcc, 90))

    logger.info(f"  E MAE (global):           {e_mae_global:.6f} eV")
    logger.info(f"  R MAE (global):           {r_mae_global:.4f} (10^-40 cgs)")
    logger.info(f"  R Pearson (global):       {r_pearson_global:.6f}")
    logger.info(f"  Spectrum PCC mean:        {pcc_mean:.4f}")
    logger.info(f"  Spectrum PCC median:      {pcc_median:.4f}")
    logger.info(f"  Spectrum PCC std:         {pcc_std:.4f}")
    logger.info(f"  Spectrum PCC p10/p90:     {pcc_p10:.4f} / {pcc_p90:.4f}")

    # --- 3. 保存 per_molecule_metrics.csv ---
    csv_path = os.path.join(quant_dir, 'per_molecule_metrics.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("mol_id,smiles,spectrum_pcc,r_mae,e_mae\n")
        for i in range(N):
            f.write(
                f"{mol_ids[i]},{smiles[i]},"
                f"{per_mol_pcc[i]:.6f},{per_mol_r_mae[i]:.4f},"
                f"{per_mol_e_mae[i]:.6f}\n"
            )
    logger.info(f"  Saved: {csv_path}")

    # --- 4. 保存 quantitative_summary.txt ---
    summary_path = os.path.join(quant_dir, 'quantitative_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("PhysECD Quantitative Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total molecules: {N}\n\n")
        f.write("--- Discrete State Metrics (N molecules x 20 states flattened) ---\n")
        f.write(f"E MAE:            {e_mae_global:.6f} eV\n")
        f.write(f"E Pearson:        {e_pearson_global:.6f}\n")
        f.write(f"R MAE:            {r_mae_global:.4f} (10^-40 cgs)\n")
        f.write(f"R Pearson:        {r_pearson_global:.6f}\n\n")
        f.write("--- Spectrum PCC (per-molecule, 80-450 nm, sigma=0.4 eV) ---\n")
        f.write(f"Mean:             {pcc_mean:.6f}\n")
        f.write(f"Median:           {pcc_median:.6f}\n")
        f.write(f"Std:              {pcc_std:.6f}\n")
        f.write(f"P10:              {pcc_p10:.6f}\n")
        f.write(f"P25:              {pcc_p25:.6f}\n")
        f.write(f"P75:              {pcc_p75:.6f}\n")
        f.write(f"P90:              {pcc_p90:.6f}\n")
    logger.info(f"  Saved: {summary_path}")

    # --- 5. PCC 分布直方图 ---
    _plot_pcc_histogram(per_mol_pcc, quant_dir, logger)

    # --- 6. R 散点图 ---
    _plot_scatter(
        R_flat_true, R_flat_pred, r_pearson_global,
        xlabel='R_true (10^-40 cgs)', ylabel='R_pred (10^-40 cgs)',
        title=f'Rotatory Strength: Predicted vs True\nPearson r = {r_pearson_global:.4f}',
        save_path=os.path.join(quant_dir, 'R_scatter.png'),
        logger=logger
    )

    # --- 7. E 散点图 ---
    _plot_scatter(
        E_flat_true, E_flat_pred, e_pearson_global,
        xlabel='E_true (eV)', ylabel='E_pred (eV)',
        title=f'Excitation Energy: Predicted vs True\nPearson r = {e_pearson_global:.4f}',
        save_path=os.path.join(quant_dir, 'E_scatter.png'),
        logger=logger
    )

    # --- 8. 最好/最差 5 个光谱示例 ---
    _plot_best_worst_spectra(
        per_mol_pcc, mol_ids, E_pred, E_true, R_pred, R_true,
        wavelength_grid, quant_dir, logger, n=5
    )


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 确定输出目录
    ckpt_path  = Path(args.checkpoint)
    output_dir = args.output_dir or str(ckpt_path.parent / f'{args.split}_evaluation')
    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info(f"PhysECD Evaluation  [split = {args.split.upper()}]")
    logger.info("=" * 80)
    logger.info(f"Device:     {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data dir:   {args.data_dir}")
    logger.info(f"Split:      {args.split}")
    logger.info(f"Output dir: {output_dir}")

    # 加载 checkpoint
    logger.info("\nLoading checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get('config')
    if config is None:
        logger.warning("Checkpoint does not contain 'config', using default parameters...")
        config = {
            'num_features': 128,
            'max_l': 3,
            'num_blocks': 3,
            'num_radial': 32,
            'cutoff': 50.0,
            'n_states': 20,
            'max_atomic_number': 60,
            'lambda_E': 1.0,
            'lambda_mu_vel': 0.0,
            'lambda_m': 0.0,
            'lambda_R': 1.0,
            'lambda_R_sign': 0.0,
        }
    epoch = checkpoint.get('epoch', '?')
    logger.info(f"Checkpoint from epoch {epoch}, val_loss={checkpoint.get('val_loss', 'N/A')}")

    # 构建模型并加载权重
    logger.info("\nBuilding model...")
    model, criterion = build_model_and_criterion(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model: {model.get_num_params():,} parameters")

    # 加载指定分割的数据
    logger.info(f"\nLoading {args.split} data...")
    data_path = os.path.join(args.data_dir, f'{args.split}.pt')
    if not Path(data_path).exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")
    data_list = torch.load(data_path, weights_only=False)
    loader = DataLoader(
        data_list,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    logger.info(f"Loaded {len(data_list)} {args.split} samples")

    # 指标评估
    logger.info("\n" + "=" * 80)
    logger.info(f"{args.split.capitalize()} Set Metrics")
    logger.info("=" * 80)
    metrics = evaluate(model, loader, criterion, device, logger)
    logger.info(
        f"Normalized Loss: {metrics['loss']:.4f}  "
        f"(E: {metrics['loss_E']:.4f}, "
        f"mu_vel: {metrics['loss_mu_vel']:.4f}, "
        f"m: {metrics['loss_m']:.4f}, "
        f"R: {metrics['loss_R']:.4f})"
    )
    logger.info(
        f"Raw MSE: E={metrics['loss_E_raw']:.4f}, "
        f"mu_vel={metrics['loss_mu_vel_raw']:.4f}, "
        f"m={metrics['loss_m_raw']:.4f}, "
        f"R={metrics['loss_R_raw']:.4f}"
    )
    logger.info(f"R Sign Acc: {metrics['R_sign_acc']:.4f}")

    # 批量光谱生成
    logger.info("\n" + "=" * 80)
    logger.info("Generating Spectrum Comparisons")
    logger.info("=" * 80)
    generate_test_spectra(
        model, loader, device, os.path.join(output_dir, 'spectra'), logger,
        num_samples=args.num_spectra
    )

    # 定量评估
    logger.info("\n" + "=" * 80)
    logger.info("Quantitative Evaluation")
    logger.info("=" * 80)
    logger.info(f"Collecting predictions for all {len(data_list)} {args.split} molecules...")
    collected = collect_predictions(model, loader, device)
    logger.info(f"Collected {len(collected['mol_ids'])} molecules")
    run_quantitative_evaluation(collected, output_dir, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete.")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
