"""
Plot ECD Spectrum Comparison (Predicted vs Real)
================================================
读取预测光谱和真实光谱的 CSV 文件，将两条曲线绘制在同一张图中。

Features:
- 蓝色实线：真实光谱（Real）
- 红色虚线：预测光谱（Predicted）
- 高分辨率输出（300 DPI）
- 专业出版风格：网格、基线、图例、粗体坐标轴标签

运行指令：
cd 到 PhysECD 项目根目录下，执行：
python scripts/08_plot_spectrum.py --pred_csv path/to/predicted.csv --real_csv path/to/real.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='Plot ECD spectrum comparison: predicted vs real'
    )

    parser.add_argument(
        '--pred_csv',
        type=str,
        required=True,
        help='预测光谱 CSV 文件路径（需包含 Wavelength (nm) 和 [θ] 两列）'
    )
    parser.add_argument(
        '--real_csv',
        type=str,
        required=True,
        help='真实光谱 CSV 文件路径（需包含 Wavelength (nm) 和 [θ] 两列）'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='输出 PNG 文件路径（默认：与 pred_csv 同目录，后缀改为 _comparison.png）'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='图表标题（可选）'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='输出分辨率，单位 DPI（默认：300）'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[10, 6],
        help='图像尺寸，单位英寸（宽 高，默认：10 6）'
    )

    return parser.parse_args()


def load_spectrum_data(csv_path):
    """
    从 CSV 文件加载光谱数据。

    Args:
        csv_path: CSV 文件路径

    Returns:
        wavelength: [N] 波长数组（nm）
        ecd: [N] 摩尔椭圆度数组（deg·cm²/dmol）
    """
    df = pd.read_csv(csv_path)

    if 'Wavelength (nm)' not in df.columns or '[θ]' not in df.columns:
        raise ValueError(
            f"CSV 文件必须包含 'Wavelength (nm)' 和 '[θ]' 两列，"
            f"当前列名：{list(df.columns)}"
        )

    wavelength = df['Wavelength (nm)'].values
    ecd = df['[θ]'].values

    return wavelength, ecd


def plot_spectrum_comparison(
    wavelength, pred_spectrum, real_spectrum,
    output_path, title=None, dpi=300, figsize=(10, 6)
):
    """
    将预测光谱和真实光谱绘制在同一张图中。

    Args:
        wavelength: [N] 波长数组（nm），两条曲线共用
        pred_spectrum: [N] 预测光谱（deg·cm²/dmol）
        real_spectrum: [N] 真实光谱（deg·cm²/dmol）
        output_path: 输出 PNG 文件路径
        title: 图表标题（可选）
        dpi: 输出分辨率
        figsize: 图像尺寸（宽, 高），单位英寸
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # 真实光谱：蓝色实线
    ax.plot(wavelength, real_spectrum, 'b-', linewidth=2.0, label='Real', alpha=0.8)
    # 预测光谱：红色虚线
    ax.plot(wavelength, pred_spectrum, 'r--', linewidth=2.0, label='Predicted', alpha=0.8)
    # 基线 y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('[θ] (deg·cm²/dmol)', fontsize=12, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(wavelength.min(), wavelength.max())
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved comparison plot to: {output_path}")
    plt.close(fig)


def main():
    """主流程：加载两条光谱，对齐波长，绘制对比图。"""
    args = parse_args()

    print("=" * 80)
    print("ECD Spectrum Comparison Plotting")
    print("=" * 80)

    # 加载预测光谱
    print(f"\n[1/3] Loading predicted spectrum...")
    pred_csv = Path(args.pred_csv)
    if not pred_csv.exists():
        raise FileNotFoundError(f"预测光谱文件不存在：{pred_csv}")
    pred_wavelength, pred_ecd = load_spectrum_data(pred_csv)
    print(f"  Loaded {len(pred_wavelength)} points from {pred_csv}")

    # 加载真实光谱
    print(f"\n[2/3] Loading real spectrum...")
    real_csv = Path(args.real_csv)
    if not real_csv.exists():
        raise FileNotFoundError(f"真实光谱文件不存在：{real_csv}")
    real_wavelength, real_ecd = load_spectrum_data(real_csv)
    print(f"  Loaded {len(real_wavelength)} points from {real_csv}")

    # 波长对齐：若两者波长不同，对预测光谱进行插值
    if not np.allclose(pred_wavelength, real_wavelength, atol=0.1):
        print(f"  Warning: wavelength grids differ, interpolating predicted spectrum onto real grid...")
        pred_ecd = np.interp(real_wavelength, pred_wavelength, pred_ecd)
        wavelength = real_wavelength
    else:
        wavelength = real_wavelength

    print(f"  Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} nm")

    # 确定输出路径
    if args.output_path is None:
        output_path = pred_csv.with_name(pred_csv.stem + '_comparison.png')
    else:
        output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 绘制对比图
    print(f"\n[3/3] Generating comparison plot...")
    print(f"  Figure size: {args.figsize[0]} x {args.figsize[1]} inches")
    print(f"  Resolution: {args.dpi} DPI")

    plot_spectrum_comparison(
        wavelength=wavelength,
        pred_spectrum=pred_ecd,
        real_spectrum=real_ecd,
        output_path=output_path,
        title=args.title,
        dpi=args.dpi,
        figsize=tuple(args.figsize)
    )

    print("\n" + "=" * 80)
    print("Plotting Complete!")
    print("=" * 80)
    print(f"\nOutput file: {output_path}")
    print(f"Resolution: {args.dpi} DPI")
    print("=" * 80)


if __name__ == "__main__":
    main()
