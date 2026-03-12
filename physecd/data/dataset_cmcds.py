"""
CSV Parser and CMCDS Dataset
=============================
This module provides utilities to parse the CMCDS_DATASET.csv file and align
the data with Gaussian log files.

The CSV file format:
- One molecule occupies 3 rows:
  Row 1: Excitation energies (eV)
  Row 2: Rotatory Strength [R(velocity)] with varying exponent units
  Row 3: Wavelengths (nm)
- Each row has 20 excited states (columns 4-23)
"""

import re
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple


class CMCDSCSVParser:
    """Parser for CMCDS_DATASET.csv file."""

    def __init__(self, csv_path: str):
        """
        Initialize CSV parser.

        Args:
            csv_path: Path to CMCDS_DATASET.csv file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV file
        self.df = pd.read_csv(csv_path)
        self._validate_csv()

    def _validate_csv(self):
        """Validate CSV file structure."""
        required_columns = ['ID', 'smiles', 'ECD Transition Parameters']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check that we have 20 excited state columns
        excited_state_cols = [col for col in self.df.columns if col.startswith('Excited State_')]
        if len(excited_state_cols) != 20:
            raise ValueError(f"Expected 20 excited state columns, found {len(excited_state_cols)}")

    def _parse_rotatory_strength_unit(self, unit_str: str) -> float:
        """
        Parse the unit exponent from rotatory strength parameter string.

        Examples:
            'Rotatory Strength [R(velocity)] (cgs(10**-40 erg-esu-cm/Gauss))' -> 1e-40
            'Rotatory Strength [R(velocity)] (cgs(10**-41 erg-esu-cm/Gauss))' -> 1e-41
            'Rotatory Strength [R(velocity)] (cgs(10**-42 erg-esu-cm/Gauss))' -> 1e-42

        Args:
            unit_str: The parameter description string

        Returns:
            Scaling factor (e.g., 1e-40)
        """
        # Match pattern like "10**-40" or "10**-41"
        pattern = r'10\*\*(-?\d+)'
        match = re.search(pattern, unit_str)

        if match:
            exponent = int(match.group(1))
            return 10.0 ** exponent
        else:
            # Default to 1e-40 if not specified
            return 1e-40

    def parse_molecule(self, mol_id: int) -> Dict[str, torch.Tensor]:
        """
        Parse data for a single molecule from CSV.

        Args:
            mol_id: Molecule ID

        Returns:
            Dictionary containing:
                - 'y_E': [20] excitation energies (eV)
                - 'y_R': [20] rotatory strengths (in 10^-40 cgs units)
                - 'smiles': SMILES string
                - 'mol_id': molecule ID
        """
        # Find the three rows for this molecule
        mol_rows = self.df[self.df['ID'] == mol_id]

        if len(mol_rows) == 0:
            raise ValueError(f"Molecule ID {mol_id} not found in CSV")

        if len(mol_rows) != 3:
            raise ValueError(f"Expected 3 rows for molecule {mol_id}, found {len(mol_rows)}")

        # Extract the three parameter rows
        rows_list = mol_rows.to_dict('records')

        # Identify which row is which by checking the 'ECD Transition Parameters' column
        energy_row = None
        rotatory_row = None
        wavelength_row = None

        for row in rows_list:
            param_name = row['ECD Transition Parameters']
            if 'Excitation energies' in param_name:
                energy_row = row
            elif 'Rotatory Strength' in param_name:
                rotatory_row = row
            elif 'Wavelengths' in param_name or 'Wavelegths' in param_name:  # Note: typo in CSV
                wavelength_row = row

        if energy_row is None or rotatory_row is None:
            raise ValueError(f"Cannot find required parameter rows for molecule {mol_id}")

        # Extract excited state columns
        excited_state_cols = [f'Excited State_{i}' for i in range(1, 21)]

        # Extract excitation energies
        energies = [energy_row[col] for col in excited_state_cols]
        y_E = torch.tensor(energies, dtype=torch.float32)

        # Extract rotatory strengths (all in 10^-40 cgs units)
        rotatory_strengths = [rotatory_row[col] for col in excited_state_cols]
        y_R = torch.tensor(rotatory_strengths, dtype=torch.float32)

        # Get SMILES
        smiles = energy_row['smiles']

        return {
            'y_E': y_E,
            'y_R': y_R,
            'smiles': smiles,
            'mol_id': mol_id
        }

    def get_all_molecule_ids(self) -> List[int]:
        """
        Get list of all unique molecule IDs in the CSV.

        Returns:
            List of molecule IDs
        """
        return sorted(self.df['ID'].unique().tolist())


if __name__ == "__main__":
    # Test the CSV parser
    csv_path = "/Users/jiangyi/Desktop/ECD光谱预测大模型课题文件/PhysECD/Raw_data_gaussian_input_output_V1/CMCDS_DATASET.csv"

    parser = CMCDSCSVParser(csv_path)

    # Test on molecules 1, 2, 3
    for mol_id in [1, 2, 3]:
        try:
            data = parser.parse_molecule(mol_id)
            print(f"\nMolecule {mol_id}:")
            print(f"  SMILES: {data['smiles'][:50]}...")
            print(f"  Excitation energies (first 5): {data['y_E'][:5].numpy()}")
            print(f"  Rotatory strengths (first 5): {data['y_R'][:5].numpy()}")
            print(f"  Energy range: {data['y_E'].min():.4f} - {data['y_E'].max():.4f} eV")
            print(f"  R range: {data['y_R'].min():.4e} - {data['y_R'].max():.4e}")
        except Exception as e:
            print(f"Error parsing molecule {mol_id}: {e}")

    # Print total number of molecules
    mol_ids = parser.get_all_molecule_ids()
    print(f"\nTotal molecules in CSV: {len(mol_ids)}")
    print(f"ID range: {min(mol_ids)} - {max(mol_ids)}")
