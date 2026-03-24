"""
Multi-task prediction heads for atomic and molecular properties.

Implements 5 prediction heads with proper SE(3) equivariance constraints:
1. Excitation energies (global scalar)
2. Atomic charges (atomic scalar)
3. Velocity electric dipoles (atomic polar vector, 1o)
4. Magnetic dipoles (atomic pseudo vector, 1e)
5. Transition currents (atomic polar vector, 1o)
"""

import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import global_mean_pool

from .modules import MLP


class MultiTaskHeads(nn.Module):
    """
    Multi-task prediction heads for PhysECD.

    Args:
        num_features: Number of input scalar features (default: 128)
        irreps_T: Input irreps for tensor features
        n_states: Number of excited states to predict (default: 20)

    Outputs:
        E_pred: [Batch_size, 20] - excitation energies
        q_A: [N_total, 20] - atomic transition charges
        mu_A_vel: [N_total, 20, 3] - atomic velocity electric dipole moments
        m_A: [N_total, 20, 3] - atomic magnetic dipole moments
        v_A: [N_total, 20, 3] - atomic transition current velocities
    """

    def __init__(self, num_features=128, irreps_T=None, n_states=20):
        super().__init__()

        self.num_features = num_features
        self.n_states = n_states

        if irreps_T is None:
            # Default irreps if not provided
            irreps_T = o3.Irreps([
                (num_features, (1, -1)),  # 1o: polar vectors
                (num_features, (2, 1))    # 2e: pseudo tensors
            ])
        self.irreps_T = irreps_T

        # Head 1: Excitation energies (global scalar property)
        # Pre-pooling MLP to process scalar features
        self.E_pre_pool = MLP(
            size=(num_features, num_features, num_features),
            act='swish'
        )
        # Final linear layer to predict n_states energies
        self.E_head = nn.Linear(num_features, n_states)

        # Head 2: Atomic transition charges (atomic scalar property)
        self.q_head = nn.Sequential(
            MLP(size=(num_features, num_features), act='swish'),
            nn.Linear(num_features, n_states)
        )

        # Head 3: Velocity electric dipole moments (atomic polar vector, 1o)
        # First project tensor features to 1o irreps
        irreps_1o = o3.Irreps([(num_features, (1, -1))])
        self.mu_vel_pre = o3.Linear(
            irreps_T,
            irreps_1o,
            internal_weights=True,
            shared_weights=True
        )
        # Then project to n_states copies of 1o vectors
        self.mu_vel_head = o3.Linear(
            irreps_1o,
            o3.Irreps([(n_states, (1, -1))]),
            internal_weights=True,
            shared_weights=True
        )

        # Head 4: Magnetic dipole moments (atomic pseudo vector, 1e)
        # First project tensor features to 1e irreps
        irreps_1e = o3.Irreps([(num_features, (1, 1))])
        self.m_pre = o3.Linear(
            irreps_T,
            irreps_1e,
            internal_weights=True,
            shared_weights=True
        )
        # Then project to n_states copies of 1e vectors
        self.m_head = o3.Linear(
            irreps_1e,
            o3.Irreps([(n_states, (1, 1))]),
            internal_weights=True,
            shared_weights=True
        )

        # Head 5: Transition current velocities (atomic polar vector, 1o)
        # Reuse irreps_1o from electric dipole
        self.v_pre = o3.Linear(
            irreps_T,
            irreps_1o,
            internal_weights=True,
            shared_weights=True
        )
        # Then project to n_states copies of 1o vectors
        self.v_head = o3.Linear(
            irreps_1o,
            o3.Irreps([(n_states, (1, -1))]),
            internal_weights=True,
            shared_weights=True
        )

    def forward(self, S, T, batch):
        """
        Forward pass through all prediction heads.

        Args:
            S: [N_total, num_features] - scalar features from backbone
            T: [N_total, irreps_T.dim] - tensor features from backbone
            batch: [N_total] - batch indices

        Returns:
            E_pred: [Batch_size, n_states] - excitation energies (guaranteed > 0)
            q_A: [N_total, n_states] - atomic charges
            mu_A_vel: [N_total, n_states, 3] - velocity electric dipoles
            m_A: [N_total, n_states, 3] - magnetic dipoles
            v_A: [N_total, n_states, 3] - transition currents
        """

        # Head 1: Excitation energies (global pooling required)
        S_processed = self.E_pre_pool(S)  # [N_total, num_features]
        S_pooled = global_mean_pool(S_processed, batch)  # [Batch_size, num_features]
        E_raw = self.E_head(S_pooled)  # [Batch_size, n_states]

        # Apply Softplus + offset to ensure E_pred > 0
        # Softplus(x) = log(1 + exp(x)), smooth approximation of ReLU
        # Offset = 3.0 ensures E_pred >= 3.0 eV (physical constraint)
        E_pred = torch.nn.functional.softplus(E_raw) + 3.0  # [Batch_size, n_states]

        # Head 2: Atomic charges (no pooling, atomic-level)
        q_A = self.q_head(S)  # [N_total, n_states]

        # Head 3: Velocity electric dipole moments
        # Output shape: [N_total, n_states * 3] (flattened)
        mu_vel_flat = self.mu_vel_head(self.mu_vel_pre(T))  # [N_total, n_states * 3]
        # Reshape to [N_total, n_states, 3]
        mu_A_vel = mu_vel_flat.reshape(-1, self.n_states, 3)

        # Head 4: Magnetic dipole moments
        # Output shape: [N_total, n_states * 3] (flattened)
        m_flat = self.m_head(self.m_pre(T))  # [N_total, n_states * 3]
        # Reshape to [N_total, n_states, 3]
        m_A = m_flat.reshape(-1, self.n_states, 3)

        # Head 5: Transition current velocities
        # Output shape: [N_total, n_states * 3] (flattened)
        v_flat = self.v_head(self.v_pre(T))  # [N_total, n_states * 3]
        # Reshape to [N_total, n_states, 3]
        v_A = v_flat.reshape(-1, self.n_states, 3)

        return E_pred, q_A, mu_A_vel, m_A, v_A
