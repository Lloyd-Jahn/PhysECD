"""
Generate Enantiomer Data for CMCDS Dataset
===========================================
This script generates enantiomer (mirror image) data for all molecules in the dataset
by applying geometric and vector symmetry transformations.

Transformation: Reflection across the XY plane
- Coordinates: (x, y, z) → (x, y, -z)
- Electric dipole (polar vector): (μx, μy, μz) → (μx, μy, -μz)
- Magnetic dipole (axial vector): (mx, my, mz) → (-mx, -my, mz)
- Rotatory strength: R → -R
- Excitation energy: unchanged

This doubles the dataset size without requiring new DFT calculations.
"""

import sys
from pathlib import Path
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import argparse
import pandas as pd

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate enantiomer data for CMCDS dataset')

    parser.add_argument(
        '--input_dir',
        type=str,
        default='/home/data/jiangyi/PhysECD/data/processed',
        help='Directory containing original .pt files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/data/jiangyi/PhysECD/data/processed_with_enantiomers',
        help='Output directory for expanded .pt files'
    )
    parser.add_argument(
        '--csv_output',
        type=str,
        default='/home/data/jiangyi/PhysECD/data/CMCDS_DATASET_with_enantiomers.csv',
        help='Output path for expanded CSV file'
    )
    parser.add_argument(
        '--original_csv',
        type=str,
        default='/home/data/jiangyi/Raw_data_gaussian_input_output_V1/CMCDS_DATASET.csv',
        help='Path to original CMCDS_DATASET.csv (for wavelength data)'
    )

    return parser.parse_args()


def generate_enantiomer(data: Data) -> Data:
    """
    Generate enantiomer data by applying reflection across XY plane.

    Scientific principles:
    1. Coordinates (x, y, z) → (x, y, -z)
    2. Electric dipole (polar vector): (μx, μy, μz) → (μx, μy, -μz)
    3. Magnetic dipole (axial vector): (mx, my, mz) → (-mx, -my, mz)
    4. Rotatory strength: R → -R
    5. Excitation energy: unchanged

    Args:
        data: Original PyG Data object

    Returns:
        Enantiomer PyG Data object
    """
    # Clone the original data
    enantiomer = Data()

    # 1. Transform coordinates: (x, y, z) → (x, y, -z)
    pos_enantiomer = data.pos.clone()
    pos_enantiomer[:, 2] = -pos_enantiomer[:, 2]  # Invert Z coordinate

    # 2. Transform velocity electric dipole moment (polar vector)
    #    (μx, μy, μz) → (μx, μy, -μz)
    y_mu_vel_enantiomer = data.y_mu_vel.clone()
    y_mu_vel_enantiomer[:, 2] = -y_mu_vel_enantiomer[:, 2]  # Invert Z component

    # 3. Transform magnetic dipole moment (axial vector/pseudovector)
    #    (mx, my, mz) → (-mx, -my, mz)
    y_m_enantiomer = data.y_m.clone()
    y_m_enantiomer[:, 0] = -y_m_enantiomer[:, 0]  # Invert X component
    y_m_enantiomer[:, 1] = -y_m_enantiomer[:, 1]  # Invert Y component
    # Z component remains unchanged

    # 4. Transform rotatory strength: R → -R
    y_R_enantiomer = -data.y_R.clone()

    # 5. Excitation energy remains unchanged
    y_E_enantiomer = data.y_E.clone()

    # Atomic numbers remain unchanged
    z_enantiomer = data.z.clone()

    # Create enantiomer Data object
    enantiomer.z = z_enantiomer
    enantiomer.pos = pos_enantiomer
    enantiomer.y_E = y_E_enantiomer
    enantiomer.y_mu_vel = y_mu_vel_enantiomer
    enantiomer.y_m = y_m_enantiomer
    enantiomer.y_R = y_R_enantiomer
    enantiomer.smiles = data.smiles  # SMILES remains the same (represents connectivity)
    enantiomer.mol_id = -data.mol_id  # Use negative ID to indicate enantiomer

    return enantiomer


def verify_transformation(original: Data, enantiomer: Data):
    """
    Verify that the enantiomer transformation is correct by checking
    that μ·m changes sign.

    For original: R = μ·m
    For enantiomer: R' = μ'·m' = -μ·m = -R

    Args:
        original: Original molecule data
        enantiomer: Enantiomer molecule data
    """
    # Compute dot product for original
    mu_dot_m_original = (original.y_mu_vel * original.y_m).sum(dim=1)

    # Compute dot product for enantiomer
    mu_dot_m_enantiomer = (enantiomer.y_mu_vel * enantiomer.y_m).sum(dim=1)

    # Check that they have opposite signs
    ratio = mu_dot_m_enantiomer / (mu_dot_m_original + 1e-10)

    # Should be close to -1
    expected_ratio = -1.0
    max_error = torch.abs(ratio - expected_ratio).max().item()

    return max_error


def expand_dataset(data_list, verify=True):
    """
    Expand dataset by generating enantiomers for all molecules.

    Args:
        data_list: List of original PyG Data objects
        verify: Whether to verify transformations

    Returns:
        Expanded list containing both originals and enantiomers
    """
    expanded_data = []
    max_errors = []

    for data in tqdm(data_list, desc="  Generating enantiomers"):
        # Add original molecule
        expanded_data.append(data)

        # Generate and add enantiomer
        enantiomer = generate_enantiomer(data)
        expanded_data.append(enantiomer)

        # Verify transformation
        if verify:
            error = verify_transformation(data, enantiomer)
            max_errors.append(error)

    if verify and len(max_errors) > 0:
        avg_error = sum(max_errors) / len(max_errors)
        max_error = max(max_errors)
        print(f"  Verification: avg error = {avg_error:.2e}, max error = {max_error:.2e}")
        if max_error > 0.01:
            print(f"  WARNING: Large transformation error detected!")

    return expanded_data


def load_wavelengths_from_csv(csv_path):
    """
    Load wavelength data from original CMCDS_DATASET.csv.

    Args:
        csv_path: Path to original CSV file

    Returns:
        Dictionary mapping mol_id → list of 20 wavelength values
    """
    df = pd.read_csv(csv_path)
    excited_state_cols = [f'Excited State_{i}' for i in range(1, 21)]
    wavelength_map = {}

    for mol_id in df['ID'].unique():
        mol_rows = df[df['ID'] == mol_id]
        for _, row in mol_rows.iterrows():
            param = row['ECD Transition Parameters']
            if 'Wavelengths' in param or 'Wavelegths' in param:  # Handle typo in CSV
                wavelengths = [row[col] for col in excited_state_cols]
                wavelength_map[int(mol_id)] = wavelengths
                break

    return wavelength_map


def generate_expanded_csv(train_data, val_data, test_data, output_path, wavelength_map):
    """
    Generate expanded CSV file with both original and enantiomer data.

    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        output_path: Path to save CSV file
        wavelength_map: Dictionary mapping mol_id → wavelength list
    """
    all_data = train_data + val_data + test_data

    rows = []
    missing_wavelengths = 0
    for data in tqdm(all_data, desc="  Creating CSV rows"):
        mol_id = data.mol_id
        smiles = data.smiles

        # For enantiomers (negative mol_id), use the original mol_id to look up wavelengths
        original_mol_id = abs(mol_id)

        # Row 1: Excitation energies
        energy_row = {
            'ID': mol_id,
            'smiles': smiles,
            'ECD Transition Parameters': 'Excitation energies (eV)'
        }
        for i, energy in enumerate(data.y_E.tolist(), 1):
            energy_row[f'Excited State_{i}'] = energy
        rows.append(energy_row)

        # Row 2: Rotatory strengths (in 10^-40 cgs units)
        rotatory_row = {
            'ID': mol_id,
            'smiles': smiles,
            'ECD Transition Parameters': 'Rotatory Strength [R(velocity)] (cgs(10**-40 erg-esu-cm/Gauss))'
        }
        for i, R in enumerate(data.y_R.tolist(), 1):
            rotatory_row[f'Excited State_{i}'] = R
        rows.append(rotatory_row)

        # Row 3: Wavelengths (same for original and enantiomer)
        wavelength_row = {
            'ID': mol_id,
            'smiles': smiles,
            'ECD Transition Parameters': 'Wavelegths (nm)'
        }
        if original_mol_id in wavelength_map:
            for i, wl in enumerate(wavelength_map[original_mol_id], 1):
                wavelength_row[f'Excited State_{i}'] = wl
        else:
            missing_wavelengths += 1
            for i in range(1, 21):
                wavelength_row[f'Excited State_{i}'] = None
        rows.append(wavelength_row)

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved CSV with {len(all_data)} molecules ({len(rows)} rows)")
    if missing_wavelengths > 0:
        print(f"  WARNING: {missing_wavelengths} molecules missing wavelength data")


def main():
    """Main pipeline for generating enantiomer data."""
    args = parse_args()

    print("=" * 80)
    print("Enantiomer Data Generation")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original datasets
    print("\n[1/5] Loading original datasets...")
    input_dir = Path(args.input_dir)

    train_data = torch.load(input_dir / 'train.pt', weights_only=False)
    val_data = torch.load(input_dir / 'val.pt', weights_only=False)
    test_data = torch.load(input_dir / 'test.pt', weights_only=False)

    print(f"  Loaded from: {input_dir}")
    print(f"    - train.pt: {len(train_data)} molecules")
    print(f"    - val.pt: {len(val_data)} molecules")
    print(f"    - test.pt: {len(test_data)} molecules")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} molecules")

    # Expand datasets by generating enantiomers
    print("\n[2/5] Generating enantiomers...")
    print("  Train set:")
    train_expanded = expand_dataset(train_data, verify=True)
    print("  Validation set:")
    val_expanded = expand_dataset(val_data, verify=True)
    print("  Test set:")
    test_expanded = expand_dataset(test_data, verify=True)

    print(f"\n  Expanded dataset sizes:")
    print(f"    - Train: {len(train_data)} → {len(train_expanded)} molecules (2x)")
    print(f"    - Val: {len(val_data)} → {len(val_expanded)} molecules (2x)")
    print(f"    - Test: {len(test_data)} → {len(test_expanded)} molecules (2x)")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} → {len(train_expanded) + len(val_expanded) + len(test_expanded)} molecules")

    # Save expanded datasets
    print("\n[3/5] Saving expanded datasets...")
    torch.save(train_expanded, output_dir / 'train.pt')
    torch.save(val_expanded, output_dir / 'val.pt')
    torch.save(test_expanded, output_dir / 'test.pt')
    print(f"  Saved to: {output_dir}")
    print(f"    - train.pt: {len(train_expanded)} samples")
    print(f"    - val.pt: {len(val_expanded)} samples")
    print(f"    - test.pt: {len(test_expanded)} samples")

    # Load wavelength data from original CSV
    print("\n[4/5] Loading wavelength data from original CSV...")
    wavelength_map = load_wavelengths_from_csv(args.original_csv)
    print(f"  Loaded wavelengths for {len(wavelength_map)} molecules")

    # Generate expanded CSV
    print("\n[5/5] Generating expanded CSV file...")
    generate_expanded_csv(train_expanded, val_expanded, test_expanded, args.csv_output, wavelength_map)
    print(f"  Saved to: {args.csv_output}")

    # Print final statistics
    print("\n" + "=" * 80)
    print("Enantiomer Generation Complete!")
    print("=" * 80)
    print(f"\nDataset expanded from {len(train_data) + len(val_data) + len(test_data)} to {len(train_expanded) + len(val_expanded) + len(test_expanded)} molecules")
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'train.pt'}")
    print(f"  - {output_dir / 'val.pt'}")
    print(f"  - {output_dir / 'test.pt'}")
    print(f"  - {args.csv_output}")

    # Verification example
    print("\n" + "=" * 80)
    print("Verification Example")
    print("=" * 80)
    original = train_data[0]
    enantiomer = train_expanded[1]  # First enantiomer

    print(f"\nOriginal molecule (ID={original.mol_id}):")
    print(f"  First atom position: {original.pos[0].numpy()}")
    print(f"  First excited state:")
    print(f"    - Energy: {original.y_E[0]:.4f} eV")
    print(f"    - μ (electric dipole): {original.y_mu_vel[0].numpy()}")
    print(f"    - m (magnetic dipole): {original.y_m[0].numpy()}")
    print(f"    - R (rotatory strength): {original.y_R[0]:.4e} (10^-40 cgs)")

    print(f"\nEnantiomer (ID={enantiomer.mol_id}):")
    print(f"  First atom position: {enantiomer.pos[0].numpy()}")
    print(f"  First excited state:")
    print(f"    - Energy: {enantiomer.y_E[0]:.4f} eV (unchanged)")
    print(f"    - μ (electric dipole): {enantiomer.y_mu_vel[0].numpy()} (Z inverted)")
    print(f"    - m (magnetic dipole): {enantiomer.y_m[0].numpy()} (X,Y inverted)")
    print(f"    - R (rotatory strength): {enantiomer.y_R[0]:.4e} (10^-40 cgs, sign flipped)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
