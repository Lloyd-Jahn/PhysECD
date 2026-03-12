"""
Plot ECD Spectrum from CSV
===========================
This script reads ECD spectrum data from CSV and generates plots.

Features:
- High-resolution output (300 DPI)
- Professional styling with grid and baseline
- Clear axis labels and formatting
- Smooth curves with appropriate colors
"""

"""
运行指令：
cd到PhysECD目录下，执行以下命令：
/home/jiangyi/.conda/envs/ecd_pred/bin/python /home/data/jiangyi/PhysECD/scripts/08_plot_spectrum.py --csv_path csv文件路径
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot ECD spectrum from CSV file')

    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to CSV file containing spectrum data'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for PNG file (default: same as CSV with .png extension)'
    )
    parser.add_argument(
        '--color',
        type=str,
        default='darkblue',
        help='Line color (default: darkblue)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output resolution in DPI (default: 300)'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[10, 6],
        help='Figure size in inches (width height, default: 10 6)'
    )

    return parser.parse_args()


def load_spectrum_data(csv_path):
    """
    Load spectrum data from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        wavelength: [N] wavelength values in nm
        [θ]: [N] molar ellipticity values in deg·cm^2/dmol
    """
    df = pd.read_csv(csv_path)

    # Check required columns
    if 'Wavelength (nm)' not in df.columns or '[θ]' not in df.columns:
        raise ValueError(f"CSV must contain 'Wavelength (nm)' and '[θ]' columns")

    wavelength = df['Wavelength (nm)'].values
    ecd = df['[θ]'].values

    return wavelength, ecd


def plot_spectrum(wavelength, ecd, output_path, title=None, color='darkblue', dpi=300, figsize=(10, 6)):
    """
    Create publication-quality ECD spectrum plot.

    Args:
        wavelength: [N] wavelength values in nm
        [θ]: [N] molar ellipticity values in deg·cm^2/dmol
        output_path: Path to save PNG file
        title: Plot title (optional)
        color: Line color
        dpi: Output resolution
        figsize: Figure size (width, height) in inches
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Plot spectrum
    ax.plot(wavelength, ecd, color=color, linewidth=2.0)

    # Add horizontal line at y=0 (baseline for Cotton effect)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.7, label='Baseline')

    # Set axis labels
    ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('[θ] (deg·cm²/dmol)', fontsize=14, fontweight='bold')

    # Set axis limits
    ax.set_xlim(wavelength.min(), wavelength.max())

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Set title
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")

    # Close figure to free memory
    plt.close(fig)


def main():
    """Main pipeline for spectrum plotting."""
    args = parse_args()

    print("=" * 80)
    print("ECD Spectrum Plotting")
    print("=" * 80)

    # Load data
    print(f"\n[1/2] Loading spectrum data...")
    print(f"  CSV file: {args.csv_path}")

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    wavelength, ecd = load_spectrum_data(csv_path)

    print(f"  Loaded {len(wavelength)} spectral points")
    print(f"  Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} nm")
    print(f"  ECD range: {ecd.min():.4e} - {ecd.max():.4e} deg·cm^2/dmol")

    # Determine output path
    if args.output_path is None:
        output_path = csv_path.with_suffix('.png')
    else:
        output_path = Path(args.output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create plot
    print(f"\n[2/2] Generating plot...")
    print(f"  Color: {args.color}")
    print(f"  Figure size: {args.figsize[0]} x {args.figsize[1]} inches")
    print(f"  Resolution: {args.dpi} DPI")

    plot_spectrum(
        wavelength=wavelength,
        ecd=ecd,
        output_path=output_path,
        color=args.color,
        dpi=args.dpi,
        figsize=tuple(args.figsize)
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Plotting Complete!")
    print("=" * 80)
    print(f"\nOutput file: {output_path}")
    print(f"Resolution: {args.dpi} DPI")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
