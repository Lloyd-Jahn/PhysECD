"""
数据分布分析脚本

分析训练集与测试集的数据分布差异，包含两个任务：
  1. Morgan 指纹相似度分析：
     找出测试集中光谱 PCC 最高的 top-N 个分子，
     计算其 Morgan 指纹与训练集所有分子的 Tanimoto 相似度，
     判断高 PCC 是否由"记忆"训练集分子导致。
  2. E / R 真值分布直方图：
     比较训练集、验证集、测试集在激发能（E）和旋转强度（R）上的分布差异。

运行方式（从项目根目录）：
  python scripts/09_distribution_analysis.py
  python scripts/09_distribution_analysis.py --top_n 50 --pcc_csv checkpoints/test_spectra/quantitative/per_molecule_metrics.csv
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# 将项目根目录加入 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# RDKit 导入
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit 未安装，将跳过 Morgan 指纹分析任务。")


# ──────────────────────────────────────────────
# 命令行参数
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='分析训练/验证/测试集的数据分布差异')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed_with_enantiomers',
        help='数据集目录（默认：data/processed_with_enantiomers）'
    )
    parser.add_argument(
        '--pcc_csv',
        type=str,
        default=None,
        help='per_molecule_metrics.csv 路径（含 mol_id, smiles, spectrum_pcc 列）。'
             '若未指定，脚本将自动在常见路径中搜索。'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=50,
        help='取 PCC 最高的前 N 个测试集分子进行指纹相似度分析（默认：50）'
    )
    parser.add_argument(
        '--morgan_radius',
        type=int,
        default=2,
        help='Morgan 指纹半径（默认：2）'
    )
    parser.add_argument(
        '--morgan_bits',
        type=int,
        default=2048,
        help='Morgan 指纹位数（默认：2048）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='distribution_analysis',
        help='输出目录（默认：distribution_analysis）'
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 数据加载工具
# ──────────────────────────────────────────────

def load_split(pt_path):
    """
    加载 .pt 文件（PyG Data 对象列表）。
    返回 smiles 列表、mol_id 列表、y_E numpy 数组 [N, 20]、y_R numpy 数组 [N, 20]。
    """
    data_list = torch.load(pt_path, weights_only=False)
    smiles_list = []
    mol_id_list = []
    E_list = []
    R_list = []

    for data in data_list:
        smiles_list.append(data.smiles)
        mol_id_list.append(int(data.mol_id))
        # y_E / y_R 可能已经是 [20] 或需要 reshape
        E_list.append(data.y_E.numpy().flatten())   # [20]
        R_list.append(data.y_R.numpy().flatten())   # [20]

    return {
        'smiles':   smiles_list,
        'mol_ids':  mol_id_list,
        'E':        np.array(E_list),   # [N, 20]
        'R':        np.array(R_list),   # [N, 20]
    }


def load_pcc_csv(pcc_csv_path):
    """
    读取 per_molecule_metrics.csv，返回 {mol_id: {'smiles': ..., 'pcc': ...}} 字典。
    """
    import csv
    records = {}
    with open(pcc_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mol_id = int(row['mol_id'])
            records[mol_id] = {
                'smiles': row['smiles'],
                'pcc':    float(row['spectrum_pcc']),
            }
    return records


def find_pcc_csv(project_root):
    """搜索 per_molecule_metrics.csv。"""
    path = "checkpoints/test_evaluation/quantitative/per_molecule_metrics.csv"
    return path


# ──────────────────────────────────────────────
# 任务 1：Morgan 指纹相似度分析
# ──────────────────────────────────────────────

def compute_morgan_fp(smiles, radius=2, n_bits=2048):
    """将 SMILES 转换为手性感知 Morgan 指纹（位向量）。失败返回 None。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=True,
    )
    return generator.GetFingerprint(mol)


def _compute_similarities_for_group(records, train_fps, train_valid_idx, train_data):
    """
    给定一组测试分子记录，批量计算每个分子与训练集的最大 Tanimoto 相似度。
    返回 results 列表，每项包含 mol_id、pcc、max_tanimoto、最近邻训练分子信息。
    """
    results = []
    for mol_id, info in records:
        test_fp = compute_morgan_fp(info['smiles'])
        if test_fp is None:
            print(f"    警告：mol_id={mol_id} SMILES 解析失败，跳过。")
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_sim  = max(similarities)
        best_idx = np.argmax(similarities)
        best_train_idx = train_valid_idx[best_idx]
        results.append({
            'test_mol_id':       mol_id,
            'test_smiles':       info['smiles'],
            'test_pcc':          info['pcc'],
            'max_tanimoto':      max_sim,
            'best_train_mol_id': train_data['mol_ids'][best_train_idx],
            'best_train_smiles': train_data['smiles'][best_train_idx],
        })
    return results


def _write_group_table(f, results, group_label):
    """在汇总文件中写一组分子的统计表格。"""
    sims = [r['max_tanimoto'] for r in results]
    f.write(f"--- {group_label} （共 {len(results)} 个分子）---\n")
    f.write(f"  Mean   = {np.mean(sims):.4f}\n")
    f.write(f"  Median = {np.median(sims):.4f}\n")
    f.write(f"  Max    = {np.max(sims):.4f}\n")
    f.write(f"  Min    = {np.min(sims):.4f}\n")
    f.write(f"  >= 0.9 : {sum(s >= 0.9 for s in sims)}\n")
    f.write(f"  >= 0.8 : {sum(s >= 0.8 for s in sims)}\n")
    f.write(f"  >= 0.7 : {sum(s >= 0.7 for s in sims)}\n\n")
    f.write(f"{'Rank':<6}{'Test mol_id':<14}{'PCC':<10}"
            f"{'Max Tanimoto':<16}{'Best train mol_id':<20}"
            f"Train SMILES (first 60 chars)\n")
    f.write("-" * 120 + "\n")
    for rank, r in enumerate(results, 1):
        f.write(
            f"{rank:<6}{r['test_mol_id']:<14}{r['test_pcc']:<10.4f}"
            f"{r['max_tanimoto']:<16.4f}{r['best_train_mol_id']:<20}"
            f"{r['best_train_smiles'][:60]}\n"
        )
    f.write("\n")


def _plot_bar(ax, results, title, xlabel):
    """在给定 Axes 上绘制 Tanimoto 条形图，颜色编码 PCC。"""
    ranks = list(range(1, len(results) + 1))
    sims  = [r['max_tanimoto'] for r in results]
    pccs  = [r['test_pcc'] for r in results]

    norm = plt.Normalize(min(pccs), max(pccs))
    cmap = plt.cm.RdYlGn
    bars = ax.bar(ranks, sims, edgecolor='white', linewidth=0.5, alpha=0.85)
    for bar, pcc in zip(bars, pccs):
        bar.set_facecolor(cmap(norm(pcc)))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.01).set_label('Spectrum PCC', fontsize=9)

    for thresh, color, label in [(0.9, 'red', 'Tanimoto = 0.9'),
                                  (0.7, 'orange', 'Tanimoto = 0.7')]:
        ax.axhline(thresh, color=color, linestyle='--', linewidth=1.2, alpha=0.8, label=label)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Max Tanimoto similarity to training set', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    return sims


def morgan_fingerprint_analysis(train_data, test_pcc_records, top_n,
                                 radius, n_bits, output_dir):
    """
    分别对测试集 PCC 最高和最低的 top_n 个分子，计算其手性感知 Morgan 指纹
    与训练集所有分子的最大 Tanimoto 相似度，并输出对比图。

    输出：
      - morgan_similarity_summary.txt      文字汇总（top + bottom 两组）
      - morgan_top{N}_similarity.png       top-N 条形图
      - morgan_bottom{N}_similarity.png    bottom-N 条形图
      - morgan_similarity_comparison.png  两组分布对比直方图
    """
    print("\n" + "=" * 60)
    print(f"任务 1：Morgan 指纹相似度分析（top/bottom-{top_n} PCC 分子 vs 训练集）")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 按 PCC 排序，取最高和最低各 top_n 个
    sorted_records = sorted(test_pcc_records.items(), key=lambda x: x[1]['pcc'], reverse=True)
    top_records    = sorted_records[:top_n]
    bottom_records = sorted_records[-top_n:]   # PCC 最低，顺序从低到高
    print(f"  Top-{top_n}    PCC range: {top_records[-1][1]['pcc']:.4f} ~ {top_records[0][1]['pcc']:.4f}")
    print(f"  Bottom-{top_n} PCC range: {bottom_records[0][1]['pcc']:.4f} ~ {bottom_records[-1][1]['pcc']:.4f}")

    # 计算训练集所有 Morgan 指纹（手性感知，只算一次）
    print(f"  计算训练集 {len(train_data['smiles'])} 个分子的 Morgan 指纹（radius={radius}, bits={n_bits}, useChirality=True）...")
    train_fps = []
    train_valid_idx = []
    for i, smi in enumerate(train_data['smiles']):
        fp = compute_morgan_fp(smi, radius, n_bits)
        if fp is not None:
            train_fps.append(fp)
            train_valid_idx.append(i)
    print(f"  训练集有效指纹：{len(train_fps)}/{len(train_data['smiles'])}")

    # 计算两组相似度
    print(f"  计算 top-{top_n} 分子的 Tanimoto 相似度...")
    top_results = _compute_similarities_for_group(
        top_records, train_fps, train_valid_idx, train_data)

    print(f"  计算 bottom-{top_n} 分子的 Tanimoto 相似度...")
    bottom_results = _compute_similarities_for_group(
        bottom_records, train_fps, train_valid_idx, train_data)
    # 让 bottom 按 PCC 从低到高排列，直觉上更清晰
    bottom_results = list(reversed(bottom_results))

    # ---- 保存文字汇总 ----
    summary_path = os.path.join(output_dir, 'morgan_similarity_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Morgan Fingerprint Similarity Analysis (useChirality=True)\n")
        f.write(f"Morgan radius={radius}, bits={n_bits}\n")
        f.write("=" * 120 + "\n\n")
        _write_group_table(f, top_results,    f"Top-{top_n} PCC test molecules vs. training set")
        _write_group_table(f, bottom_results, f"Bottom-{top_n} PCC test molecules vs. training set")
    print(f"  已保存：{summary_path}")

    top_sims    = [r['max_tanimoto'] for r in top_results]
    bottom_sims = [r['max_tanimoto'] for r in bottom_results]

    # ---- 绘图 1：top-N 条形图 ----
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    _plot_bar(ax, top_results,
              title=f'Morgan Fingerprint Similarity: Top-{top_n} PCC Test Molecules vs. Training Set',
              xlabel=f'Test molecule rank (by PCC, descending, rank 1 = best)')
    plt.tight_layout()
    top_bar_path = os.path.join(output_dir, f'morgan_top{top_n}_similarity.png')
    plt.savefig(top_bar_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存：{top_bar_path}")

    # ---- 绘图 2：bottom-N 条形图 ----
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    _plot_bar(ax, bottom_results,
              title=f'Morgan Fingerprint Similarity: Bottom-{top_n} PCC Test Molecules vs. Training Set',
              xlabel=f'Test molecule rank (by PCC, ascending, rank 1 = worst)')
    plt.tight_layout()
    bot_bar_path = os.path.join(output_dir, f'morgan_bottom{top_n}_similarity.png')
    plt.savefig(bot_bar_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存：{bot_bar_path}")

    # ---- 绘图 3：两组分布对比直方图 ----
    bins = np.linspace(0, 1, 26)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    for ax, sims, label, color in [
        (axes[0], top_sims,    f'Top-{top_n}',    '#2196F3'),
        (axes[1], bottom_sims, f'Bottom-{top_n}', '#F44336'),
    ]:
        ax.hist(sims, bins=bins, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(np.mean(sims),   color='black',  linestyle='--', linewidth=1.5,
                   label=f'Mean = {np.mean(sims):.3f}')
        ax.axvline(np.median(sims), color='gray',   linestyle=':',  linewidth=1.5,
                   label=f'Median = {np.median(sims):.3f}')
        for thresh, c in [(0.9, 'red'), (0.7, 'orange')]:
            ax.axvline(thresh, color=c, linestyle=':', linewidth=1.0, alpha=0.6)
        ax.set_xlabel('Max Tanimoto similarity to nearest training molecule', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{label} PCC Test Molecules\n'
                     f'(>= 0.9: {sum(s >= 0.9 for s in sims)}, '
                     f'>= 0.8: {sum(s >= 0.8 for s in sims)}, '
                     f'>= 0.7: {sum(s >= 0.7 for s in sims)})',
                     fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(f'Max Tanimoto Similarity Distribution: Top vs. Bottom {top_n} PCC Test Molecules',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    cmp_path = os.path.join(output_dir, 'morgan_similarity_comparison.png')
    plt.savefig(cmp_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存：{cmp_path}")

    # 控制台摘要
    def _print_stats(label, sims):
        print(f"  {label}：均值={np.mean(sims):.4f}，中位数={np.median(sims):.4f}，"
              f"≥0.9: {sum(s>=0.9 for s in sims)}, "
              f"≥0.8: {sum(s>=0.8 for s in sims)}, "
              f"≥0.7: {sum(s>=0.7 for s in sims)}")

    print(f"\n  Tanimoto 相似度统计对比：")
    _print_stats(f"Top-{top_n}   ", top_sims)
    _print_stats(f"Bottom-{top_n}", bottom_sims)


# ──────────────────────────────────────────────
# 任务 2：E / R 真值分布直方图
# ──────────────────────────────────────────────

def plot_distribution_histograms(train_data, val_data, test_data, output_dir):
    """
    绘制训练集/验证集/测试集的 E（激发能）和 R（旋转强度）真值分布直方图。

    每个分子有 20 个激发态，将所有激发态的 E / R 展平后绘制。
    输出两张图：
      - distribution_E.png   激发能分布
      - distribution_R.png   旋转强度分布（全范围 + 截断范围各一行）
    """
    print("\n" + "=" * 60)
    print("任务 2：E / R 真值分布直方图")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 展平所有激发态
    E_train = train_data['E'].flatten()
    E_val   = val_data['E'].flatten()
    E_test  = test_data['E'].flatten()

    R_train = train_data['R'].flatten()
    R_val   = val_data['R'].flatten()
    R_test  = test_data['R'].flatten()

    n_train_mol = len(train_data['smiles'])
    n_val_mol   = len(val_data['smiles'])
    n_test_mol  = len(test_data['smiles'])

    print(f"  训练集：{n_train_mol} 个分子，{len(E_train)} 个激发态")
    print(f"  验证集：{n_val_mol} 个分子，{len(E_val)} 个激发态")
    print(f"  测试集：{n_test_mol} 个分子，{len(E_test)} 个激发态")

    colors = {'Train': '#2196F3', 'Val': '#FF9800', 'Test': '#4CAF50'}
    splits = [
        ('Train', E_train, R_train),
        ('Val',   E_val,   R_val),
        ('Test',  E_test,  R_test),
    ]

    # ────────────── 图 1：激发能 E ──────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    fig.suptitle('Ground-truth Excitation Energy E Distribution (all 20 states flattened)', fontsize=14, fontweight='bold', y=1.01)

    # 确定统一 bins 范围
    E_all = np.concatenate([E_train, E_val, E_test])
    E_bins = np.linspace(E_all.min(), E_all.max(), 60)

    for ax, (name, E_vals, _) in zip(axes, splits):
        color = colors[name]
        ax.hist(E_vals, bins=E_bins, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.axvline(np.mean(E_vals),   color='red',    linestyle='--', linewidth=1.5,
                   label=f'Mean={np.mean(E_vals):.3f}')
        ax.axvline(np.median(E_vals), color='darkred', linestyle=':',  linewidth=1.5,
                   label=f'Median={np.median(E_vals):.3f}')
        ax.set_title(f'{name} ({len(E_vals):,} states)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Excitation Energy E (eV)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    e_path = os.path.join(output_dir, 'distribution_E.png')
    plt.savefig(e_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存：{e_path}")

    # ────────────── 图 2：旋转强度 R ──────────────
    # R 的分布通常有长尾，画两行：全范围 + 截断到 [-500, 500]
    R_all = np.concatenate([R_train, R_val, R_test])
    R_clip_lo, R_clip_hi = -500.0, 500.0
    R_bins_full = np.linspace(R_all.min(), R_all.max(), 80)
    R_bins_clip = np.linspace(R_clip_lo, R_clip_hi, 80)

    fig = plt.figure(figsize=(15, 10), dpi=150)
    fig.suptitle('Ground-truth Rotatory Strength R Distribution (all 20 states flattened, unit: 10^-40 cgs)',
                 fontsize=14, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for col, (name, _, R_vals) in enumerate(splits):
        color = colors[name]

        # 第一行：全范围
        ax_full = fig.add_subplot(gs[0, col])
        ax_full.hist(R_vals, bins=R_bins_full, color=color, alpha=0.8,
                     edgecolor='white', linewidth=0.3)
        ax_full.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax_full.axvline(np.mean(R_vals),   color='red',    linestyle='--', linewidth=1.5,
                        label=f'Mean={np.mean(R_vals):.1f}')
        ax_full.axvline(np.median(R_vals), color='darkred', linestyle=':',  linewidth=1.5,
                        label=f'Median={np.median(R_vals):.1f}')
        ax_full.set_title(f'{name} (full range)', fontsize=11, fontweight='bold')
        ax_full.set_xlabel('R (10^-40 cgs)', fontsize=10)
        ax_full.set_ylabel('Count', fontsize=10)
        ax_full.legend(fontsize=8)
        ax_full.grid(True, alpha=0.3, linestyle='--')

        # 第二行：截断到 [-500, 500]
        ax_clip = fig.add_subplot(gs[1, col])
        R_clipped = R_vals[(R_vals >= R_clip_lo) & (R_vals <= R_clip_hi)]
        clip_ratio = len(R_clipped) / len(R_vals) * 100
        ax_clip.hist(R_clipped, bins=R_bins_clip, color=color, alpha=0.8,
                     edgecolor='white', linewidth=0.3)
        ax_clip.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax_clip.axvline(np.mean(R_clipped),   color='red',    linestyle='--', linewidth=1.5,
                        label=f'Mean={np.mean(R_clipped):.1f}')
        ax_clip.axvline(np.median(R_clipped), color='darkred', linestyle=':',  linewidth=1.5,
                        label=f'Median={np.median(R_clipped):.1f}')
        ax_clip.set_title(f'{name} (clipped to ±500, {clip_ratio:.1f}% of data)',
                          fontsize=11, fontweight='bold')
        ax_clip.set_xlabel('R (10^-40 cgs)', fontsize=10)
        ax_clip.set_ylabel('Count', fontsize=10)
        ax_clip.legend(fontsize=8)
        ax_clip.grid(True, alpha=0.3, linestyle='--')

    r_path = os.path.join(output_dir, 'distribution_R.png')
    plt.savefig(r_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存：{r_path}")

    # ────────────── 图 3：叠加对比图（E 和 R 各一张）──────────────
    # 归一化频率直方图，方便视觉比较三个分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # E 叠加
    ax = axes[0]
    for name, E_vals, _ in splits:
        ax.hist(E_vals, bins=E_bins, density=True, color=colors[name],
                alpha=0.5, edgecolor='none', label=name)
    ax.set_xlabel('Excitation Energy E (eV)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Excitation Energy E: Normalized Distribution Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # R 叠加（截断范围）
    ax = axes[1]
    for name, _, R_vals in splits:
        R_clipped = R_vals[(R_vals >= R_clip_lo) & (R_vals <= R_clip_hi)]
        ax.hist(R_clipped, bins=R_bins_clip, density=True, color=colors[name],
                alpha=0.5, edgecolor='none', label=name)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Rotatory Strength R (10^-40 cgs, clipped to ±500)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Rotatory Strength R: Normalized Distribution (clipped to ±500)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    overlay_path = os.path.join(output_dir, 'distribution_overlay.png')
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存：{overlay_path}")

    # 打印统计摘要
    print("\n  --- 激发能 E 统计 ---")
    for name, E_vals, _ in splits:
        print(f"    {name}：均值={np.mean(E_vals):.4f} eV，"
              f"中位数={np.median(E_vals):.4f} eV，"
              f"std={np.std(E_vals):.4f} eV，"
              f"范围=[{E_vals.min():.3f}, {E_vals.max():.3f}]")

    print("\n  --- 旋转强度 R 统计 ---")
    for name, _, R_vals in splits:
        print(f"    {name}：均值={np.mean(R_vals):.2f}，"
              f"中位数={np.median(R_vals):.2f}，"
              f"std={np.std(R_vals):.2f}，"
              f"范围=[{R_vals.min():.1f}, {R_vals.max():.1f}]  (10⁻⁴⁰ cgs)")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PhysECD 数据分布分析")
    print("=" * 60)

    # ---- 加载三个分割的数据 ----
    data_dir = Path(args.data_dir)
    print(f"\n加载数据集：{data_dir}")

    train_pt = data_dir / 'train.pt'
    val_pt   = data_dir / 'val.pt'
    test_pt  = data_dir / 'test.pt'

    for p in [train_pt, val_pt, test_pt]:
        if not p.exists():
            raise FileNotFoundError(f"找不到数据文件：{p}")

    print("  加载 train.pt ...")
    train_data = load_split(str(train_pt))
    print(f"  训练集：{len(train_data['smiles'])} 个分子")

    print("  加载 val.pt ...")
    val_data = load_split(str(val_pt))
    print(f"  验证集：{len(val_data['smiles'])} 个分子")

    print("  加载 test.pt ...")
    test_data = load_split(str(test_pt))
    print(f"  测试集：{len(test_data['smiles'])} 个分子")

    # ---- 任务 2：E / R 分布直方图（不依赖评估结果，先跑）----
    plot_distribution_histograms(train_data, val_data, test_data, output_dir)

    # ---- 任务 1：Morgan 指纹相似度分析 ----
    if not RDKIT_AVAILABLE:
        print("\n警告：RDKit 不可用，跳过 Morgan 指纹分析任务。")
    else:
        # 确定 per_molecule_metrics.csv 路径
        pcc_csv_path = args.pcc_csv
        if pcc_csv_path is None:
            pcc_csv_path = find_pcc_csv(project_root)
        if pcc_csv_path is None or not Path(pcc_csv_path).exists():
            print("\n警告：未找到 per_molecule_metrics.csv，跳过 Morgan 指纹分析任务。")
            print("  请先运行 04_evaluate.py 生成评估结果，或使用 --pcc_csv 参数指定路径。")
        else:
            print(f"\n使用 PCC 数据：{pcc_csv_path}")
            pcc_records = load_pcc_csv(pcc_csv_path)
            print(f"  共 {len(pcc_records)} 条记录")

            morgan_fingerprint_analysis(
                train_data    = train_data,
                test_pcc_records = pcc_records,
                top_n         = args.top_n,
                radius        = args.morgan_radius,
                n_bits        = args.morgan_bits,
                output_dir    = output_dir,
            )

    print("\n" + "=" * 60)
    print(f"分析完成。所有输出已保存至：{output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
