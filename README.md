# PhysECD: Physics-Driven SE(3) Framework for ECD Spectrum Prediction

A deep learning framework combining SE(3) equivariant neural networks with physics-based aggregation layers for predicting Electronic Circular Dichroism (ECD) spectra from molecular 3D structures.

## Project Overview

This project aims to predict ECD spectra from chiral organic molecules using a physics-driven approach. Unlike existing methods that treat spectral prediction as a pure sequence/image generation task, PhysECD integrates first-principles physics into the model architecture.

### Key Features

- **SE(3) Equivariant Architecture**: Uses SE(3) Graph Neural Networks to preserve rotational and translational symmetry
- **Physics-Based Aggregation**: Predicts atomic-level transition contributions and aggregates them using rigorous physical formulas
- **Multi-Task Learning**: Jointly predicts excitation energies, dipole moments, and rotatory strengths
- **Direct CMCDS Training**: Optimized for the CMCDS dataset without requiring pretraining

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- pandas
- tqdm

### Setup

```bash
# Clone the repository
git clone <repository_url>
cd PhysECD

# Install dependencies (requirements.txt to be created)
pip install torch torch_geometric pandas tqdm
```

## Data Preparation (✓ Completed)

The data preparation pipeline has been successfully implemented and tested.

### Dataset Statistics

- **Total molecules**: 10,887
- **Training set**: 8,709 molecules (80%)
- **Validation set**: 1,088 molecules (10%)
- **Test set**: 1,090 molecules (10%)

### Molecular Properties

- **Atoms per molecule**: 13-127 (mean: 43.1)
- **Excitation energies**: 2.46-11.98 eV (mean: 6.62 ± 1.11 eV)
- **Rotatory strengths**: -2.91e-39 to 5.11e-39 (1e-40 cgs units)

### Running Data Preparation

```bash
cd PhysECD
python scripts/01_prepare_data.py
```

The script will:
1. Parse CMCDS_DATASET.csv for labels (E, R)
2. Extract atomic coordinates from .gjf files
3. Extract dipole moments from .log files
4. Create PyTorch Geometric Data objects
5. Split into train/val/test sets
6. Save as .pt files in `data/processed/`

### Data Format

Each PyG Data object contains:

```python
Data(
    z=[N_atoms],              # Atomic numbers (int64)
    pos=[N_atoms, 3],         # 3D coordinates (float32)
    y_E=[20],                 # Excitation energies (float32)
    y_mu_vel=[20, 3],         # Velocity electric dipole moments (float32)
    y_m=[20, 3],              # Magnetic dipole moments (float32)
    y_R=[20],                 # Rotatory strengths (float32)
    smiles=str,               # SMILES string
    mol_id=int                # Molecule ID
)
```

## Project Structure

```
PhysECD/
├── README.md                   # This file
├── requirements.txt            # Python dependencies (to be created)
├── configs/                    # Configuration files
│   └── finetune_cmcds.yaml    # CMCDS training config (to be created)
│
├── data/                       # Data directory
│   └── processed/             # Processed PyG datasets
│       ├── train.pt           # Training set (8,709 molecules)
│       ├── val.pt             # Validation set (1,088 molecules)
│       └── test.pt            # Test set (1,090 molecules)
│
├── physecd/                    # Core source code
│   ├── __init__.py
│   ├── data/                   # Data processing modules
│   │   ├── __init__.py
│   │   ├── parser.py          # Gaussian file parser (✓ Completed)
│   │   └── dataset_cmcds.py   # CMCDS CSV parser (✓ Completed)
│   │
│   ├── models/                 # Neural network architectures (To be implemented)
│   │   ├── __init__.py
│   │   ├── se3_backbone.py    # SE(3) equivariant backbone
│   │   ├── heads.py           # Prediction heads
│   │   └── physecd_model.py   # Complete model
│   │
│   ├── physics/                # Physics-based layers (To be implemented)
│   │   ├── __init__.py
│   │   ├── aggregation.py     # Physics aggregation layer
│   │   ├── rendering.py       # Spectrum rendering
│   │   └── loss.py            # Physics-constrained loss functions
│   │
│   └── utils/                  # Utilities (To be implemented)
│       ├── __init__.py
│       ├── metrics.py         # Evaluation metrics
│       └── logger.py          # Training logger
│
├── scripts/                    # Execution scripts
│   ├── 01_prepare_data.py     # Data preparation (✓ Completed)
│   ├── 02_pretrain.py         # Pretraining (Skipped for now)
│   ├── 03_finetune.py         # CMCDS training (To be implemented)
│   └── 04_evaluate.py         # Evaluation (To be implemented)
│
└── notebooks/                  # Jupyter notebooks (To be created)
    ├── 1_data_exploration.ipynb
    ├── 2_spectra_plot.ipynb
    └── 3_interpretability.ipynb
```

## Model Architecture (To be Implemented)

### 1. Input
- `pos`: [N_atoms, 3] - DFT-optimized 3D coordinates
- `z`: [N_atoms] - Atomic numbers

### 2. Backbone
- SE(3) Transformer (e3nn-based)

### 3. Prediction Heads
- **Global**: Excitation energies E [20]
- **Atomic-level**:
  - Transition charges q_A [N_atoms, 20]
  - Local electric dipoles μ_A [N_atoms, 20, 3]
  - Local magnetic dipoles m_A [N_atoms, 20, 3]
  - Local transition currents v_A [N_atoms, 20, 3]

### 4. Physics Aggregation (Parameter-free)
- Coordinate centering
- Charge/current conservation
- Global dipole moment calculation
- Rotatory strength: R = μ_total · m_total

### 5. Loss Function
- Multi-task MSE on E, μ_total, m_total, R
- Physical constraint regularization

## Next Steps

1. ✅ **Data Preparation** - COMPLETED
   - [x] Implement Gaussian file parser
   - [x] Implement CSV parser
   - [x] Create data preparation script
   - [x] Process CMCDS dataset

2. **Model Implementation** - IN PROGRESS
   - [ ] Implement SE(3) backbone
   - [ ] Implement prediction heads
   - [ ] Implement physics aggregation layer
   - [ ] Implement loss functions

3. **Training Pipeline**
   - [ ] Create training script
   - [ ] Implement evaluation metrics
   - [ ] Add logging and checkpointing

4. **Evaluation and Analysis**
   - [ ] Compare with baseline methods
   - [ ] Visualize predicted vs. true spectra
   - [ ] Interpretability analysis

## Citation

(To be added upon publication)

## License

(To be determined)

## Acknowledgments

This project builds upon:
- TD-DFT calculations using Gaussian 16
- CMCDS dataset
- e3nn library for equivariant neural networks
- PyTorch Geometric for graph neural networks
