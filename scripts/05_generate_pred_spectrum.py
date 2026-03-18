"""
Generate ECD Spectrum from Model Predictions
=============================================
This script performs inference on a trained PhysECD model and generates
continuous ECD spectra through Gaussian broadening.

Physics Formula:
1. Unit conversion: R_cgs = R_au × 471.44 (in 10^-40 cgs units)
2. Energy grid: E = 1240 / λ (eV), λ ∈ [80, 450] nm
3. Gaussian broadening:
   Δε(E) = (1 / (2.297×10^1 × σ × √π)) × Σ E_i × R_cgs,i × exp[-(E - E_i)^2 / σ^2]
   Note: R_cgs is a numerical value in 10^-40 cgs units, so the normalization
   constant is 2.297×10^1 (not 2.297×10^-39 which is for absolute cgs units)
4. Convert to molar ellipticity: [θ] = Δε × 3298.2
"""

"""
运行指令：
cd到PhysECD目录下，执行以下命令：
/home/jiangyi/.conda/envs/ecd_pred/bin/python /home/data/jiangyi/PhysECD-3.17修改-公式严谨性检查+单独训练R/scripts/06_generate_pred_spectrum.py --mol_id 6283
最后的数字”6283”是分子ID，可以自己指定
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physecd.models import PhysECDModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate ECD spectrum from model predictions')

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default='data/processed_with_enantiomers/test.pt',
        help='Path to test dataset'
    )
    parser.add_argument(
        '--mol_id',
        type=int,
        required=True,
        help='Molecule ID to predict (e.g. 6283 or -6283)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ecd_pred_results',
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.4,
        help='Gaussian broadening standard deviation in eV (default: 0.3)'
    )
    parser.add_argument(
        '--wavelength_min',
        type=float,
        default=80.0,
        help='Minimum wavelength in nm (default: 80)'
    )
    parser.add_argument(
        '--wavelength_max',
        type=float,
        default=450.0,
        help='Maximum wavelength in nm (default: 450)'
    )
    parser.add_argument(
        '--wavelength_step',
        type=float,
        default=1.0,
        help='Wavelength step in nm (default: 1.0)'
    )

    return parser.parse_args()


def load_model_and_data(checkpoint_path, test_data_path):
    """
    Load trained model and test dataset.

    Args:
        checkpoint_path: Path to checkpoint file
        test_data_path: Path to test data .pt file

    Returns:
        model: Loaded PhysECDModel
        test_data: List of PyG Data objects
        config: Training configuration dict
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['config']

    # Construct model
    model = PhysECDModel(
        num_features=config['num_features'],
        max_l=config['max_l'],
        num_blocks=config['num_blocks'],
        num_radial=config['num_radial'],
        cutoff=config['cutoff'],
        n_states=config['n_states'],
        max_atomic_number=config['max_atomic_number']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Model loaded successfully")
    print(f"  Training epoch: {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")

    # Load test data
    print(f"\nLoading test data from: {test_data_path}")
    test_data = torch.load(test_data_path, weights_only=False)
    print(f"  Test set size: {len(test_data)} molecules")

    return model, test_data, config


def run_inference(model, data, device):
    """
    Run model inference on a single molecule.

    Args:
        model: PhysECDModel
        data: PyG Data object for single molecule
        device: torch device

    Returns:
        E_pred: [20] predicted excitation energies (eV)
        R_pred: [20] predicted rotatory strengths (a.u.)
    """
    from torch_geometric.data import Batch
    if getattr(data, 'batch', None) is None:
        data = Batch.from_data_list([data])

    model = model.to(device)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    E_pred = output['E_pred'][0].cpu().numpy()  # [20]
    R_pred = output['R_pred'][0].cpu().numpy()  # [20]

    return E_pred, R_pred


def gaussian_broadening(E_pred, R_pred_au, wavelength_grid, sigma=0.3):
    """
    Apply Gaussian broadening to generate continuous ECD spectrum.

    Physics Formula:
    1. R_cgs = R_au × 471.44 (in 10^-40 cgs units)
    2. E_grid = 1240 / λ (eV)
    3. Δε(E) = (1 / (2.297×10^1 × σ × √π)) × Σ E_i × R_cgs,i × exp[-(E - E_i)^2 / σ^2]
       Note: R_cgs is a numerical value in 10^-40 cgs units, so the normalization
       constant is 2.297×10^1 (not 2.297×10^-39 which is for absolute cgs units)
    4. [θ] = Δε × 3298.2

    Args:
        E_pred: [20] excitation energies in eV
        R_pred_au: [20] rotatory strengths in atomic units
        wavelength_grid: [N] wavelength values in nm
        sigma: Gaussian broadening width in eV

    Returns:
        molar_ellipticity: [N] molar ellipticity values
    """
    # Step 1: Convert R from a.u. to cgs units (10^-40 erg-esu-cm/Gauss)
    R_cgs = R_pred_au * 471.44  # Result is in units of 10^-40 cgs

    # Step 2: Convert wavelength grid to energy grid
    # E (eV) = 1240 / λ (nm)
    E_grid = 1240.0 / wavelength_grid  # [N] energies in eV

    # Step 3: Gaussian broadening to compute Δε
    # Δε(E) = (1 / (2.296×10^1 × σ × √π)) × Σ E_i × R_cgs,i × exp[-(E - E_i)^2 / σ^2]
    # Note: Since R_cgs is in units of 10^-40 cgs (numerical value without the 10^-40 factor),
    # the normalization constant must be: 2.296×10^-39 × 10^40 = 2.296×10^1

    # Normalization constant (for R in 10^-40 cgs units)
    norm_constant = 2.296e1 * sigma * np.sqrt(np.pi)

    # Initialize Δε array
    delta_epsilon = np.zeros_like(E_grid)

    # Sum over all 20 excited states
    for i in range(len(E_pred)):
        # Gaussian function: exp[-(E - E_i)^2 / σ^2]
        gaussian = np.exp(-((E_grid - E_pred[i]) / sigma) ** 2)

        # Add contribution from this state
        delta_epsilon += E_pred[i] * R_cgs[i] * gaussian

    # Apply normalization
    delta_epsilon /= norm_constant

    # Step 4: Convert Δε to [θ] (molar ellipticity)
    # [θ] (molar ellipticity) = Δε × 3298.2
    molar_ellipticity = delta_epsilon * 3298.2

    return molar_ellipticity


def save_spectrum_csv(wavelength_grid, molar_ellipticity, output_path):
    """
    Save spectrum to CSV file.

    Args:
        wavelength_grid: [N] wavelength values in nm
        molar_ellipticity: [N] ECD values in deg·cm^2·dmol^-1
        output_path: Path to save CSV file
    """
    df = pd.DataFrame({
        'Wavelength (nm)': wavelength_grid,
        '[θ]': molar_ellipticity
    })

    df.to_csv(output_path, index=False)
    print(f"  Saved spectrum to: {output_path}")


def main():
    """Main pipeline for spectrum generation."""
    args = parse_args()

    print("=" * 80)
    print("ECD Spectrum Generation from Model Predictions")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model and data
    print("\n[1/4] Loading model and test data...")
    model, test_data, config = load_model_and_data(args.checkpoint, args.test_data)

    # Select molecule by mol_id
    print(f"\n[2/4] Searching for molecule with ID={args.mol_id}...")
    data = None
    for sample in test_data:
        mid = sample.mol_id.item() if hasattr(sample.mol_id, 'item') else int(sample.mol_id)
        if mid == args.mol_id:
            data = sample
            break
    if data is None:
        raise ValueError(f"Molecule ID {args.mol_id} not found in test set")

    mol_id = args.mol_id
    print(f"  Molecule ID: {mol_id}")
    print(f"  Number of atoms: {data.z.shape[0]}")
    print(f"  SMILES: {data.smiles}")

    # Run inference
    print(f"\n[3/4] Running model inference...")
    E_pred, R_pred_au = run_inference(model, data, device)

    print(f"  Predicted excitation energies (first 5): {E_pred[:5]}")
    print(f"  Predicted rotatory strengths (first 5): {R_pred_au[:5]}")
    print(f"  Energy range: {E_pred.min():.4f} - {E_pred.max():.4f} eV")
    print(f"  R range: {R_pred_au.min():.4e} - {R_pred_au.max():.4e} a.u.")

    # Generate spectrum through Gaussian broadening
    print(f"\n[4/4] Generating continuous spectrum...")
    print(f"  Wavelength range: {args.wavelength_min} - {args.wavelength_max} nm")
    print(f"  Wavelength step: {args.wavelength_step} nm")
    print(f"  Gaussian broadening σ: {args.sigma} eV")

    # Create wavelength grid
    wavelength_grid = np.arange(args.wavelength_min, args.wavelength_max + args.wavelength_step, args.wavelength_step)

    # Apply Gaussian broadening
    molar_ellipticity = gaussian_broadening(E_pred, R_pred_au, wavelength_grid, sigma=args.sigma)

    print(f"  Generated {len(wavelength_grid)} spectral points")
    print(f"  ECD [θ] range: {molar_ellipticity.min():.4e} - {molar_ellipticity.max():.4e} deg·cm^2/dmol")

    # Save to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{mol_id}_predicted.csv"
    save_spectrum_csv(wavelength_grid, molar_ellipticity, output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("Spectrum Generation Complete!")
    print("=" * 80)
    print(f"\nOutput file: {output_path}")
    print(f"Molecule ID: {mol_id}")
    print(f"Spectral points: {len(wavelength_grid)}")
    print(f"Wavelength range: {wavelength_grid[0]:.1f} - {wavelength_grid[-1]:.1f} nm")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
