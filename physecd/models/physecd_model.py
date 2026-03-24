"""
Complete PhysECD model integrating all components.

Combines SE(3) backbone, multi-task heads, and physics-based aggregation
into a single end-to-end model for ECD spectrum prediction.
"""

import torch
import torch.nn as nn

from .se3_backbone import SE3Backbone
from .heads import MultiTaskHeads
from ..physics.aggregation import PhysicsAggregation


class PhysECDModel(nn.Module):
    """
    Physics-driven SE(3) equivariant model for ECD spectrum prediction.

    Architecture:
        1. SE(3) Backbone: Extract equivariant features from molecular structure
        2. Multi-Task Heads: Predict atomic-level quantum properties
        3. Physics Aggregation: Aggregate to molecular properties using physics

    Args:
        num_features: Number of features per channel (default: 128)
        max_l: Maximum spherical harmonic degree (default: 2)
        num_blocks: Number of interaction blocks (default: 3)
        num_radial: Number of radial basis functions (default: 32)
        cutoff: Cutoff radius in Ångströms (default: 5.0)
        n_states: Number of excited states (default: 20)
        max_atomic_number: Maximum atomic number (default: 36)

    Input (PyG Data object):
        - z: [N_total] - atomic numbers
        - pos: [N_total, 3] - atomic positions
        - batch: [N_total] - batch indices

    Output (dict):
        - E_pred: [Batch_size, 20] - excitation energies
        - mu_total_vel: [Batch_size, 20, 3] - total velocity electric dipole moments
        - m_total: [Batch_size, 20, 3] - total magnetic dipole moments
        - R_pred: [Batch_size, 20] - rotatory strengths
    """

    def __init__(
        self,
        num_features=128,
        max_l=2,
        num_blocks=3,
        num_radial=32,
        cutoff=5.0,
        n_states=20,
        max_atomic_number=36
    ):
        super().__init__()

        self.num_features = num_features
        self.n_states = n_states

        # Component 1: SE(3) equivariant backbone
        self.backbone = SE3Backbone(
            num_features=num_features,
            max_l=max_l,
            num_blocks=num_blocks,
            num_radial=num_radial,
            cutoff=cutoff,
            max_atomic_number=max_atomic_number
        )

        # Component 2: Multi-task prediction heads
        self.heads = MultiTaskHeads(
            num_features=num_features,
            irreps_T=self.backbone.irreps_T,
            n_states=n_states
        )

        # Component 3: Physics-based aggregation (parameter-free)
        self.physics_agg = PhysicsAggregation()

    def forward(self, data):
        """
        Forward pass through complete model.

        Args:
            data: PyTorch Geometric Data object with fields:
                - z: [N_total] - atomic numbers
                - pos: [N_total, 3] - atomic positions
                - batch: [N_total] - batch indices

        Returns:
            Dictionary containing:
                - E_pred: [Batch_size, 20] - predicted excitation energies
                - mu_total_vel: [Batch_size, 20, 3] - total velocity electric dipole moments
                - m_total: [Batch_size, 20, 3] - total magnetic dipole moments
                - R_pred: [Batch_size, 20] - predicted rotatory strengths
        """

        # Step 1: Extract SE(3) equivariant features
        S, T = self.backbone(data.z, data.pos, data.batch)
        # S: [N_total, num_features] - scalar features
        # T: [N_total, irreps_T.dim] - tensor features

        # Step 2: Predict atomic-level quantum properties
        E_pred, q_A, mu_A_vel, m_A, v_A = self.heads(S, T, data.batch)
        # E_pred: [Batch_size, 20] - excitation energies
        # q_A: [N_total, 20] - atomic transition charges
        # mu_A_vel: [N_total, 20, 3] - atomic velocity electric dipoles
        # m_A: [N_total, 20, 3] - atomic magnetic dipoles
        # v_A: [N_total, 20, 3] - atomic transition currents

        # Step 3: Aggregate atomic properties to molecular properties
        mu_total_vel, m_total, R_pred = self.physics_agg(
            pos=data.pos,
            batch=data.batch,
            q_A=q_A,
            mu_A_vel=mu_A_vel,
            m_A=m_A,
            v_A=v_A,
            E_pred=E_pred
        )
        # mu_total_vel: [Batch_size, 20, 3] - total velocity electric dipole moments
        # m_total: [Batch_size, 20, 3] - total magnetic dipole moments
        # R_pred: [Batch_size, 20] - rotatory strengths

        # Return all predictions
        return {
            'E_pred': E_pred,
            'mu_total_vel': mu_total_vel,
            'm_total': m_total,
            'R_pred': R_pred
        }

    def get_num_params(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
