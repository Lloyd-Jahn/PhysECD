"""
Data Preparation Script for CMCDS Dataset
==========================================
This script processes the raw CMCDS dataset and converts it into PyTorch Geometric
format (.pt files) for training.

Workflow:
1. Parse CSV file to get molecule IDs and labels (E, R)
2. Parse Gaussian files (.gjf, .log) to get structures and dipole moments
3. Merge data and save as PyG Data objects
4. Split into train/val/test sets
"""

import sys
from pathlib import Path
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import argparse

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physecd.data.parser import GaussianParser
from physecd.data.dataset_cmcds import CMCDSCSVParser


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare CMCDS dataset for training')

    parser.add_argument(
        '--csv_path',
        type=str,
        default='/home/data/jiangyi/PhysECD/data/CMCDS_DATASET.csv',
        help='Path to CMCDS_DATASET.csv (corrected version)'
    )
    parser.add_argument(
        '--gjf_dir',
        type=str,
        default='/home/data/jiangyi/Raw_data_gaussian_input_output_V1/Raw_data_gaussian_input_output/Raw_data_gaussian_input_output/Raw_Data_ECD_gaussian_input_output/5.GJF_TD（包含优化后的3D坐标）',
        help='Directory containing .gjf files'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/home/data/jiangyi/Raw_data_gaussian_input_output_V1/Raw_data_gaussian_input_output/Raw_data_gaussian_input_output/Raw_Data_ECD_gaussian_input_output/6.ECD_LOG（含电偶极矩和磁偶极矩）',
        help='Directory containing .log files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/data/jiangyi/PhysECD/data/processed',
        help='Output directory for processed .pt files'
    )
    parser.add_argument(
        '--n_states',
        type=int,
        default=20,
        help='Number of excited states to extract'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of training set'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Ratio of validation set'
    )

    return parser.parse_args()


def process_single_molecule(mol_id, csv_parser, gaussian_parser, n_states=20):
    """
    Process a single molecule and create a PyG Data object.

    Args:
        mol_id: Molecule ID
        csv_parser: CMCDSCSVParser instance
        gaussian_parser: GaussianParser instance
        n_states: Number of excited states

    Returns:
        PyG Data object or None if processing fails
    """
    try:
        # Parse CSV data (E, R, SMILES)
        csv_data = csv_parser.parse_molecule(mol_id)

        # Parse Gaussian files (pos, z, mu_vel, m)
        gaussian_data = gaussian_parser.parse_molecule(mol_id, n_states)

        # Create PyG Data object
        data = Data(
            z=gaussian_data['z'],                    # [N_atoms] atomic numbers
            pos=gaussian_data['pos'],                # [N_atoms, 3] coordinates
            y_E=csv_data['y_E'],                     # [20] excitation energies
            y_mu_vel=gaussian_data['y_mu_vel'],      # [20, 3] velocity dipole moments
            y_m=gaussian_data['y_m'],                # [20, 3] magnetic dipole moments
            y_R=csv_data['y_R'],                     # [20] rotatory strengths (10^-40 cgs)
            smiles=csv_data['smiles'],               # SMILES string
            mol_id=mol_id                            # molecule ID
        )

        return data

    except FileNotFoundError:
        # Missing .gjf or .log file - skip this molecule
        return None
    except Exception as e:
        print(f"Error processing molecule {mol_id}: {e}")
        return None


def main():
    """Main data processing pipeline."""
    args = parse_args()

    print("=" * 80)
    print("CMCDS Dataset Preparation")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize parsers
    print("\n[1/5] Initializing parsers...")
    csv_parser = CMCDSCSVParser(args.csv_path)
    gaussian_parser = GaussianParser(args.gjf_dir, args.log_dir)
    print(f"  CSV parser initialized: {args.csv_path}")
    print(f"  Gaussian parser initialized")
    print(f"    - GJF directory: {args.gjf_dir}")
    print(f"    - LOG directory: {args.log_dir}")

    # Get all molecule IDs from CSV
    print("\n[2/5] Getting molecule IDs from CSV...")
    all_mol_ids = csv_parser.get_all_molecule_ids()
    print(f"  Found {len(all_mol_ids)} molecules in CSV (ID range: {min(all_mol_ids)}-{max(all_mol_ids)})")

    # Process all molecules
    print("\n[3/5] Processing molecules...")
    successful_data = []
    failed_ids = []

    for mol_id in tqdm(all_mol_ids, desc="  Processing"):
        data = process_single_molecule(mol_id, csv_parser, gaussian_parser, args.n_states)
        if data is not None:
            successful_data.append(data)
        else:
            failed_ids.append(mol_id)

    print(f"\n  Successfully processed: {len(successful_data)} molecules")
    print(f"  Failed/missing: {len(failed_ids)} molecules")

    if len(failed_ids) > 0:
        print(f"  Failed IDs (first 20): {failed_ids[:20]}")

    if len(successful_data) == 0:
        print("\nERROR: No molecules were successfully processed!")
        return

    # Split dataset
    print("\n[4/5] Splitting dataset...")
    n_total = len(successful_data)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val

    # Shuffle with fixed seed for reproducibility
    torch.manual_seed(42)
    indices = torch.randperm(n_total).tolist()

    train_data = [successful_data[i] for i in indices[:n_train]]
    val_data = [successful_data[i] for i in indices[n_train:n_train+n_val]]
    test_data = [successful_data[i] for i in indices[n_train+n_val:]]

    print(f"  Train: {len(train_data)} molecules ({len(train_data)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_data)} molecules ({len(val_data)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} molecules ({len(test_data)/n_total*100:.1f}%)")

    # Save datasets
    print("\n[5/5] Saving processed datasets...")
    torch.save(train_data, output_dir / 'train.pt')
    torch.save(val_data, output_dir / 'val.pt')
    torch.save(test_data, output_dir / 'test.pt')
    print(f"  Saved to: {output_dir}")
    print(f"    - train.pt: {len(train_data)} samples")
    print(f"    - val.pt: {len(val_data)} samples")
    print(f"    - test.pt: {len(test_data)} samples")

    # Print statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)

    sample_data = train_data[0]
    print(f"\nSample data structure (molecule {sample_data.mol_id}):")
    print(f"  z (atomic numbers):      shape={sample_data.z.shape}, dtype={sample_data.z.dtype}")
    print(f"  pos (coordinates):       shape={sample_data.pos.shape}, dtype={sample_data.pos.dtype}")
    print(f"  y_E (energies):          shape={sample_data.y_E.shape}, dtype={sample_data.y_E.dtype}")
    print(f"  y_mu_vel (vel dipoles):  shape={sample_data.y_mu_vel.shape}, dtype={sample_data.y_mu_vel.dtype}")
    print(f"  y_m (mag dipoles):       shape={sample_data.y_m.shape}, dtype={sample_data.y_m.dtype}")
    print(f"  y_R (rotatory strength): shape={sample_data.y_R.shape}, dtype={sample_data.y_R.dtype}")
    print(f"  smiles:                  {sample_data.smiles[:60]}...")

    # Compute statistics across dataset
    all_num_atoms = [data.z.shape[0] for data in successful_data]
    all_energies = torch.stack([data.y_E for data in successful_data])
    all_R = torch.stack([data.y_R for data in successful_data])

    print(f"\nGlobal statistics:")
    print(f"  Number of atoms per molecule:")
    print(f"    - Min: {min(all_num_atoms)}")
    print(f"    - Max: {max(all_num_atoms)}")
    print(f"    - Mean: {sum(all_num_atoms)/len(all_num_atoms):.1f}")
    print(f"  Excitation energies (eV):")
    print(f"    - Range: [{all_energies.min():.4f}, {all_energies.max():.4f}]")
    print(f"    - Mean: {all_energies.mean():.4f}")
    print(f"    - Std: {all_energies.std():.4f}")
    print(f"  Rotatory strengths (10^-40 cgs):")
    print(f"    - Range: [{all_R.min():.4e}, {all_R.max():.4e}]")
    print(f"    - Mean: {all_R.mean():.4e}")
    print(f"    - Std: {all_R.std():.4e}")

    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
