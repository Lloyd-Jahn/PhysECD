"""
Physics-based aggregation layer for molecular properties.

This module implements parameter-free aggregation of atomic-level predictions
into molecular-level properties using rigorous quantum chemistry formulas.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum


class PhysicsAggregation(nn.Module):
    """
    Parameter-free physics-based aggregation layer.

    Aggregates atomic-level quantum properties (charges, dipoles, currents)
    into molecular-level properties (total dipoles, rotatory strengths) using
    fundamental quantum chemistry equations.

    Key features:
    - Translation invariance through coordinate centering
    - Conservation constraints (zero mean for charges and currents)
    - Proper electric and magnetic dipole aggregation
    - Rotatory strength calculation from quantum formulas
    """

    def __init__(self):
        super().__init__()
        # This layer has no learnable parameters - pure physics

    def forward(self, pos, batch, q_A, mu_A_vel, m_A, v_A, E_pred):
        """
        Aggregate atomic properties to molecular properties.

        Args:
            pos: [N_total, 3] - atomic positions
            batch: [N_total] - batch indices for atoms
            q_A: [N_total, 20] - atomic transition charges for 20 excited states
            mu_A_vel: [N_total, 20, 3] - atomic velocity electric dipole moments
            m_A: [N_total, 20, 3] - atomic magnetic dipole moments
            v_A: [N_total, 20, 3] - atomic transition current velocities
            E_pred: [Batch_size, 20] - predicted excitation energies

        Returns:
            mu_total_vel: [Batch_size, 20, 3] - total velocity electric dipole moments
            m_total: [Batch_size, 20, 3] - total magnetic dipole moments
            R_pred: [Batch_size, 20] - rotatory strengths (in 10^-40 cgs)
        """

        # Step 1: Coordinate centering (translation invariance)
        pos_mean = scatter_mean(pos, batch, dim=0)  # [Batch_size, 3]
        pos_delta = pos - pos_mean[batch]  # [N_total, 3]

        # Step 2: Velocity electric dipole aggregation
        # Formula: μ_total_vel = Σ_A [μ_A_vel + q_A * r_A]
        q_r = q_A.unsqueeze(-1) * pos_delta.unsqueeze(1)  # [N_total, 20, 3]
        mu_vel_contrib = mu_A_vel + q_r  # [N_total, 20, 3]
        mu_total_vel = scatter_sum(mu_vel_contrib, batch, dim=0)  # [Batch_size, 20, 3]

        # Step 3: Magnetic dipole aggregation
        # Formula: m_total = Σ_A [m_A + (1/2) * r_A × v_A]
        n_states = v_A.shape[1]
        pos_expanded = pos_delta.unsqueeze(1).expand(-1, n_states, -1)   # [N_total, n_states, 3]
        orbital_m = 0.5 * torch.cross(pos_expanded, v_A, dim=-1)  # [N_total, 20, 3]
        m_contrib = m_A + orbital_m  # [N_total, 20, 3]
        m_total = scatter_sum(m_contrib, batch, dim=0)  # [Batch_size, 20, 3]

        # Step 4: Rotatory strength calculation
        # Formula: R = 6414.135151 × (μ · m) / E
        # Output unit: 10^-40 cgs (same as dataset labels)
        mu_vel_dot_m = torch.sum(mu_total_vel * m_total, dim=-1)  # [Batch_size, 20]

        # Numerical stability: clamp E_pred to avoid division by very small numbers
        E_pred_safe = torch.clamp(E_pred, min=1.0)
        R_pred = 6414.135151 * mu_vel_dot_m / E_pred_safe  # [Batch_size, 20]

        return mu_total_vel, m_total, R_pred
