"""
PhysECD - Physics-Driven SE(3) Framework for ECD Spectrum Prediction
====================================================================

A deep learning framework combining SE(3) equivariant neural networks
with physics-based aggregation layers for predicting Electronic Circular
Dichroism (ECD) spectra from molecular 3D structures.

Modules:
- data: Data parsing and dataset utilities
- models: Neural network architectures
- physics: Physics-based aggregation and rendering layers
- utils: Utilities for training and evaluation
"""

__version__ = "0.1.0"

from .models import PhysECDModel, SE3Backbone, MultiTaskHeads
from .physics import PhysicsAggregation, PhysECDLoss

__all__ = [
    'PhysECDModel',
    'SE3Backbone',
    'MultiTaskHeads',
    'PhysicsAggregation',
    'PhysECDLoss'
]
