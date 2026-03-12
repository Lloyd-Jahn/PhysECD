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