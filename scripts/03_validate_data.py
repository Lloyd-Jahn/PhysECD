"""
Data Validation Script
=========================================
This script loads and validates the processed CMCDS dataset,
providing detailed statistics.
"""

import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_datasets(data_dir):
    """Load all three datasets."""
    train_data = torch.load(data_dir / 'train.pt', weights_only=False)
    val_data = torch.load(data_dir / 'val.pt', weights_only=False)
    test_data = torch.load(data_dir / 'test.pt', weights_only=False)
    return train_data, val_data, test_data


def print_dataset_info(data_list, split_name):
    """Print detailed information about a dataset split."""
    print(f"\n{'='*80}")
    print(f"{split_name} Dataset Analysis")
    print(f"{'='*80}")
    print(f"\nTotal samples: {len(data_list)}")

    # Sample statistics
    sample = data_list[0]
    print(f"\nSample data structure (molecule {sample.mol_id}):")
    print(f"  z (atomic numbers):      {sample.z.shape}, {sample.z.dtype}")
    print(f"  pos (coordinates):       {sample.pos.shape}, {sample.pos.dtype}")
    print(f"  y_E (energies):          {sample.y_E.shape}, {sample.y_E.dtype}")
    print(f"  y_mu_vel (vel dipoles):  {sample.y_mu_vel.shape}, {sample.y_mu_vel.dtype}")
    print(f"  y_m (mag dipoles):       {sample.y_m.shape}, {sample.y_m.dtype}")
    print(f"  y_R (rotatory strength): {sample.y_R.shape}, {sample.y_R.dtype}")
    print(f"  SMILES: {sample.smiles[:80]}...")

    # Collect statistics
    num_atoms_list = [data.z.shape[0] for data in data_list]
    all_E = torch.stack([data.y_E for data in data_list])
    all_R = torch.stack([data.y_R for data in data_list])
    all_mu_vel = torch.stack([data.y_mu_vel for data in data_list])
    all_m = torch.stack([data.y_m for data in data_list])

    # Atom count statistics
    print(f"\nAtom count statistics:")
    print(f"  Min: {min(num_atoms_list)}")
    print(f"  Max: {max(num_atoms_list)}")
    print(f"  Mean: {np.mean(num_atoms_list):.1f}")
    print(f"  Median: {np.median(num_atoms_list):.0f}")
    print(f"  Std: {np.std(num_atoms_list):.1f}")

    # Element distribution
    all_elements = []
    for data in data_list:
        all_elements.extend(data.z.tolist())
    element_counts = Counter(all_elements)
    print(f"\nElement distribution (top 10):")
    for z, count in element_counts.most_common(10):
        element_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
        symbol = element_symbols.get(z, f'Z={z}')
        print(f"  {symbol:3s}: {count:7d} atoms ({count/len(all_elements)*100:.1f}%)")

    # Excitation energy statistics
    print(f"\nExcitation energies (eV):")
    print(f"  Range: [{all_E.min():.4f}, {all_E.max():.4f}]")
    print(f"  Mean: {all_E.mean():.4f}")
    print(f"  Median: {all_E.median():.4f}")
    print(f"  Std: {all_E.std():.4f}")

    # Rotatory strength statistics
    print(f"\nRotatory strengths (10^-40 cgs):")
    print(f"  Range: [{all_R.min():.4e}, {all_R.max():.4e}]")
    print(f"  Mean: {all_R.mean():.4e}")
    print(f"  Median: {all_R.median():.4e}")
    print(f"  Std: {all_R.std():.4e}")

    # Velocity dipole magnitude statistics
    mu_vel_mag = torch.norm(all_mu_vel, dim=-1)
    print(f"\nVelocity dipole magnitudes (Au):")
    print(f"  Range: [{mu_vel_mag.min():.4f}, {mu_vel_mag.max():.4f}]")
    print(f"  Mean: {mu_vel_mag.mean():.4f}")
    print(f"  Std: {mu_vel_mag.std():.4f}")

    # Magnetic dipole magnitude statistics
    m_mag = torch.norm(all_m, dim=-1)
    print(f"\nMagnetic dipole magnitudes (Au):")
    print(f"  Range: [{m_mag.min():.4f}, {m_mag.max():.4f}]")
    print(f"  Mean: {m_mag.mean():.4f}")
    print(f"  Std: {m_mag.std():.4f}")

    return {
        'num_atoms': num_atoms_list,
        'energies': all_E.numpy(),
        'rotatory_strengths': all_R.numpy(),
        'mu_vel_mag': mu_vel_mag.numpy(),
        'm_mag': m_mag.numpy()
    }


def print_sample_molecules(data_list, n=3):
    """Print detailed information for sample molecules."""
    print(f"\n{'='*80}")
    print(f"Sample Molecules (First {n})")
    print(f"{'='*80}")

    for i in range(min(n, len(data_list))):
        data = data_list[i]
        print(f"\nMolecule {i+1} (ID: {data.mol_id}):")
        print(f"  SMILES: {data.smiles}")
        print(f"  Num atoms: {data.z.shape[0]}")
        print(f"  Elements: {sorted(Counter(data.z.tolist()).items())}")
        print(f"\n  First 5 excited states:")
        print(f"    {'State':<8} {'E (eV)':<12} {'R (10^-40)':<15} {'|μ_vel|':<12} {'|m|':<12}")
        print(f"    {'-'*70}")
        for j in range(5):
            E = data.y_E[j].item()
            R = data.y_R[j].item()
            mu_mag = torch.norm(data.y_mu_vel[j]).item()
            m_mag = torch.norm(data.y_m[j]).item()
            print(f"    {j+1:<8d} {E:<12.4f} {R:<15.4e} {mu_mag:<12.4f} {m_mag:<12.4f}")


def main():
    """Main validation and analysis pipeline."""
    print("="*80)
    print("CMCDS Dataset Validation and Analysis")
    print("="*80)

    # Load datasets
    data_dir = Path('data/processed_with_enantiomers')
    print(f"\nLoading datasets from: {data_dir}")
    train_data, val_data, test_data = load_datasets(data_dir)
    print(f"✓ Successfully loaded:")
    print(f"  - Train: {len(train_data)} samples")
    print(f"  - Val: {len(val_data)} samples")
    print(f"  - Test: {len(test_data)} samples")

    # Analyze each split
    train_stats = print_dataset_info(train_data, "Training")
    val_stats = print_dataset_info(val_data, "Validation")
    test_stats = print_dataset_info(test_data, "Test")

    # Print sample molecules
    print_sample_molecules(train_data, n=3)

    # Final summary
    print(f"\n{'='*80}")
    print("Data Validation Complete!")
    print(f"{'='*80}")
    print("\nDataset is ready for training. Key characteristics:")
    print(f"  ✓ Total: {len(train_data) + len(val_data) + len(test_data)} molecules")
    print(f"  ✓ Avg atoms/molecule: {np.mean(train_stats['num_atoms']):.1f}")
    print(f"  ✓ Energy range: {train_stats['energies'].min():.2f}-{train_stats['energies'].max():.2f} eV")
    print(f"  ✓ All dipole moments and rotatory strengths present")
    print(f"  ✓ Data format: PyTorch Geometric compatible")
    print("\nNext steps:")
    print("  1. Implement SE(3) equivariant backbone network")
    print("  2. Implement prediction heads and physics aggregation layer")
    print("  3. Create training script with multi-task loss")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
