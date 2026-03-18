"""
Generate Real ECD Spectrum from CMCDS Dataset
==============================================
Read rotatory strengths and excitation energies from CSV and calculate molar ellipticity.

Physics Formula (from 06_generate_spectrum.py):
1. R_cgs = R_au × 471.44 (in 10^-40 cgs units)
2. E_grid = 1240 / λ (eV), λ ∈ [80, 450] nm
3. Δε(E) = (1 / (2.297×10^1 × σ × √π)) × Σ E_i × R_cgs,i × exp[-(E - E_i)^2 / σ^2]
4. [θ] = Δε × 3298.2
"""

"""
运行指令：
cd到PhysECD目录下，执行以下命令：
/home/jiangyi/.conda/envs/ecd_pred/bin/python /home/data/jiangyi/PhysECD-main/scripts/07_generate_real_spectrum.py --mol_id 6283
最后的数字”6283”是分子ID，可以自己指定
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Generate real ECD spectrum from CMCDS dataset')
    parser.add_argument('--input_csv', type=str,
                        default='/home/data/jiangyi/PhysECD-main/data/CMCDS_DATASET_with_enantiomers.csv',
                        help='Input CSV file')
    parser.add_argument('--mol_id', type=int, required=True,
                        help='Molecule ID (e.g. 6283 or -6283)')
    parser.add_argument('--output_dir', type=str, default='ecd_real_results', help='Output directory')
    parser.add_argument('--sigma', type=float, default=0.4, help='Gaussian broadening σ (eV)')
    parser.add_argument('--wavelength_min', type=float, default=80.0, help='Min wavelength (nm)')
    parser.add_argument('--wavelength_max', type=float, default=450.0, help='Max wavelength (nm)')
    parser.add_argument('--wavelength_step', type=float, default=1.0, help='Wavelength step (nm)')
    return parser.parse_args()


def gaussian_broadening(E_pred, R_cgs, wavelength_grid, sigma):
    # R_cgs is already in 10^-40 cgs units from the CSV
    E_grid = 1240.0 / wavelength_grid
    norm_constant = 2.296e1 * sigma * np.sqrt(np.pi)
    delta_epsilon = np.zeros_like(E_grid)

    for i in range(len(E_pred)):
        gaussian = np.exp(-((E_grid - E_pred[i]) / sigma) ** 2)
        delta_epsilon += E_pred[i] * R_cgs[i] * gaussian

    delta_epsilon /= norm_constant
    molar_ellipticity = delta_epsilon * 3298.2
    return molar_ellipticity


def main():
    args = parse_args()

    # Read CSV
    df = pd.read_csv(args.input_csv)

    # Find molecule by mol_id
    mol_id = args.mol_id
    mol_data = df[df['ID'] == mol_id]
    if mol_data.empty:
        # mol_id might be stored as different type, try matching
        available_ids = df['ID'].unique()
        raise ValueError(
            f"Molecule ID {mol_id} not found in CSV. "
            f"Available IDs (first 10): {available_ids[:10].tolist()}"
        )

    # Extract excitation energies
    E_row = mol_data[mol_data['ECD Transition Parameters'] == 'Excitation energies (eV)']
    E_cols = [f'Excited State_{i}' for i in range(1, 21)]
    E_pred = E_row[E_cols].values[0].astype(float)

    # Extract rotatory strengths (already in cgs 10^-40 units)
    R_row = mol_data[mol_data['ECD Transition Parameters'] == 'Rotatory Strength [R(velocity)] (cgs(10**-40 erg-esu-cm/Gauss))']
    R_cgs = R_row[E_cols].values[0].astype(float)

    # Generate spectrum
    wavelength_grid = np.arange(args.wavelength_min, args.wavelength_max + args.wavelength_step, args.wavelength_step)
    molar_ellipticity = gaussian_broadening(E_pred, R_cgs, wavelength_grid, args.sigma)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_df = pd.DataFrame({
        'Wavelength (nm)': wavelength_grid,
        '[θ]': molar_ellipticity
    })

    output_path = output_dir / f"{mol_id}_real.csv"
    output_df.to_csv(output_path, index=False)

    print(f"Generated spectrum for molecule {mol_id}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
