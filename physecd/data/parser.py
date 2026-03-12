"""
Gaussian Log Parser
===================
This module provides a parser for extracting physical quantities from Gaussian
TD-DFT calculation output files (.gjf and .log).

Key Quantities Extracted:
1. Atomic coordinates (pos) and atomic numbers (z) from .gjf files
2. Velocity electric dipole moments (mu_vel) from .log files
3. Magnetic dipole moments (m) from .log files
"""

import re
import torch
from pathlib import Path
from typing import Dict, Tuple


class GaussianParser:
    """Parser for Gaussian TD-DFT calculation files."""

    # Atomic number dictionary
    ATOMIC_NUMBERS = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    }

    def __init__(self, gjf_dir: str, log_dir: str):
        """
        Initialize parser with directory paths.

        Args:
            gjf_dir: Path to directory containing .gjf files (e.g., 5.GJF_TD)
            log_dir: Path to directory containing .log files (e.g., 6.ECD_LOG)
        """
        self.gjf_dir = Path(gjf_dir)
        self.log_dir = Path(log_dir)

        if not self.gjf_dir.exists():
            raise FileNotFoundError(f"GJF directory not found: {gjf_dir}")
        if not self.log_dir.exists():
            raise FileNotFoundError(f"LOG directory not found: {log_dir}")

    def parse_gjf_file(self, mol_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse .gjf file to extract atomic coordinates and atomic numbers.

        Args:
            mol_id: Molecule ID (e.g., 1 for molecule_1_ECD.gjf)

        Returns:
            Tuple of (pos, z):
                pos: [N_atoms, 3] FloatTensor of 3D coordinates (Angstroms)
                z: [N_atoms] LongTensor of atomic numbers
        """
        gjf_path = self.gjf_dir / f"molecule_{mol_id}_ECD.gjf"

        if not gjf_path.exists():
            raise FileNotFoundError(f"GJF file not found: {gjf_path}")

        with open(gjf_path, 'r') as f:
            lines = f.readlines()

        # Find the start of coordinate section (after "0 1" line)
        coord_start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == '0 1':
                coord_start_idx = i + 1
                break

        if coord_start_idx is None:
            raise ValueError(f"Cannot find coordinate section in {gjf_path}")

        # Extract coordinates until blank line
        atom_symbols = []
        coordinates = []

        for i in range(coord_start_idx, len(lines)):
            line = lines[i].strip()
            if not line:  # Empty line marks end of coordinates
                break

            parts = line.split()
            if len(parts) >= 4:
                atom_symbol = parts[0]
                x, y, z_coord = float(parts[1]), float(parts[2]), float(parts[3])
                atom_symbols.append(atom_symbol)
                coordinates.append([x, y, z_coord])

        if len(atom_symbols) == 0:
            raise ValueError(f"No atoms found in {gjf_path}")

        # Convert atom symbols to atomic numbers
        atomic_numbers = []
        for symbol in atom_symbols:
            if symbol not in self.ATOMIC_NUMBERS:
                raise ValueError(f"Unknown element symbol: {symbol}")
            atomic_numbers.append(self.ATOMIC_NUMBERS[symbol])

        pos = torch.tensor(coordinates, dtype=torch.float32)
        z = torch.tensor(atomic_numbers, dtype=torch.long)

        return pos, z

    def parse_log_file(self, mol_id: int, n_states: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse .log file to extract velocity dipole and magnetic dipole moments.

        Args:
            mol_id: Molecule ID (e.g., 1 for molecule_1_ECD.log)
            n_states: Number of excited states to extract (default: 20)

        Returns:
            Tuple of (mu_vel, m):
                mu_vel: [n_states, 3] FloatTensor of velocity electric dipole moments (Au)
                m: [n_states, 3] FloatTensor of magnetic dipole moments (Au)
        """
        log_path = self.log_dir / f"molecule_{mol_id}_ECD.log"

        if not log_path.exists():
            raise FileNotFoundError(f"LOG file not found: {log_path}")

        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract velocity electric dipole moments
        mu_vel = self._extract_velocity_dipole(content, n_states)

        # Extract magnetic dipole moments
        m = self._extract_magnetic_dipole(content, n_states)

        return mu_vel, m

    def _extract_velocity_dipole(self, content: str, n_states: int) -> torch.Tensor:
        """
        Extract velocity electric dipole moments from log file content.

        Pattern to match:
        Ground to excited state transition velocity dipole moments (Au):
               state          X           Y           Z        Dip. S.      Osc.
                 1         0.0268     -0.0007      0.0107      0.0008      0.0027
                 2         0.1568     -0.0441      0.0098      0.0266      0.0781
                 ...
        """
        pattern = r'Ground to excited state transition velocity dipole moments \(Au\):\s*\n\s*state\s+X\s+Y\s+Z\s+Dip\. S\.\s+Osc\.\s*\n((?:\s*\d+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+)'

        match = re.search(pattern, content)
        if not match:
            raise ValueError("Cannot find velocity dipole moments section in log file")

        data_block = match.group(1)
        lines = data_block.strip().split('\n')

        dipoles = []
        for line in lines[:n_states]:
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                dipoles.append([x, y, z])

        if len(dipoles) != n_states:
            raise ValueError(f"Expected {n_states} states, but found {len(dipoles)} in velocity dipole section")

        return torch.tensor(dipoles, dtype=torch.float32)

    def _extract_magnetic_dipole(self, content: str, n_states: int) -> torch.Tensor:
        """
        Extract magnetic dipole moments from log file content.

        Pattern to match:
        Ground to excited state transition magnetic dipole moments (Au):
               state          X           Y           Z
                 1         0.0387      0.1912     -0.0331
                 2         0.0887      0.1736      0.5919
                 ...
        """
        pattern = r'Ground to excited state transition magnetic dipole moments \(Au\):\s*\n\s*state\s+X\s+Y\s+Z\s*\n((?:\s*\d+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+)'

        match = re.search(pattern, content)
        if not match:
            raise ValueError("Cannot find magnetic dipole moments section in log file")

        data_block = match.group(1)
        lines = data_block.strip().split('\n')

        dipoles = []
        for line in lines[:n_states]:
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                dipoles.append([x, y, z])

        if len(dipoles) != n_states:
            raise ValueError(f"Expected {n_states} states, but found {len(dipoles)} in magnetic dipole section")

        return torch.tensor(dipoles, dtype=torch.float32)

    def parse_molecule(self, mol_id: int, n_states: int = 20) -> Dict[str, torch.Tensor]:
        """
        Parse both .gjf and .log files for a molecule.

        Args:
            mol_id: Molecule ID
            n_states: Number of excited states (default: 20)

        Returns:
            Dictionary containing:
                - 'pos': [N_atoms, 3] atomic coordinates
                - 'z': [N_atoms] atomic numbers
                - 'y_mu_vel': [n_states, 3] velocity dipole moments
                - 'y_m': [n_states, 3] magnetic dipole moments
        """
        pos, z = self.parse_gjf_file(mol_id)
        mu_vel, m = self.parse_log_file(mol_id, n_states)

        return {
            'pos': pos,
            'z': z,
            'y_mu_vel': mu_vel,
            'y_m': m,
            'mol_id': mol_id
        }


if __name__ == "__main__":
    # Test the parser
    gjf_dir = "/Users/jiangyi/Desktop/ECD光谱预测大模型课题文件/PhysECD/Raw_data_gaussian_input_output_V1/Raw_data_gaussian_input_output/Raw_data_gaussian_input_output/Raw_Data_ECD_gaussian_input_output/5.GJF_TD（包含优化后的3D坐标）"
    log_dir = "/Users/jiangyi/Desktop/ECD光谱预测大模型课题文件/PhysECD/Raw_data_gaussian_input_output_V1/Raw_data_gaussian_input_output/Raw_data_gaussian_input_output/Raw_Data_ECD_gaussian_input_output/6.ECD_LOG（含电偶极矩和磁偶极矩）"

    parser = GaussianParser(gjf_dir, log_dir)

    # Test on molecule 1
    try:
        data = parser.parse_molecule(1)
        print("Successfully parsed molecule 1:")
        print(f"  - Atoms: {len(data['z'])}")
        print(f"  - Coordinates shape: {data['pos'].shape}")
        print(f"  - Velocity dipole shape: {data['y_mu_vel'].shape}")
        print(f"  - Magnetic dipole shape: {data['y_m'].shape}")
        print(f"\nFirst 3 atoms:")
        for i in range(3):
            print(f"  Atom {i+1}: Z={data['z'][i].item()}, pos={data['pos'][i].numpy()}")
        print(f"\nFirst 3 excited states (velocity dipole):")
        print(data['y_mu_vel'][:3])
        print(f"\nFirst 3 excited states (magnetic dipole):")
        print(data['y_m'][:3])
    except Exception as e:
        print(f"Error: {e}")
