"""
SE(3) equivariant backbone network.
"""

import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import radius_graph

from .modules import (
    Embedding,
    Radial_Basis,
    Interaction_Block,
)


class SE3Backbone(nn.Module):
    """
    SE(3) equivariant feature extraction backbone.

    Extracts scalar and equivariant tensor features from molecular graphs
    using message passing with spherical harmonics and radial basis functions.

    Args:
        num_features: Number of features in each channel (default: 128)
        max_l: Maximum spherical harmonic degree (default: 2)
        num_blocks: Number of interaction blocks (default: 3)
        num_radial: Number of radial basis functions (default: 32)
        cutoff: Cutoff radius for neighbor search in Ångströms (default: 5.0)
        max_atomic_number: Maximum atomic number in dataset (default: 36)

    Outputs:
        S: [N_total, num_features] - scalar features
        T: [N_total, irreps_T.dim] - equivariant tensor features
    """

    def __init__(
        self,
        num_features=128,
        max_l=2,
        num_blocks=3,
        num_radial=32,
        cutoff=5.0,
        max_atomic_number=36
    ):
        super().__init__()

        self.num_features = num_features
        self.max_l = max_l
        self.num_blocks = num_blocks
        self.cutoff = cutoff

        # Define irreps for equivariant features (T)
        # 1o: polar vectors, 1e: pseudo vectors, 2e: pseudo tensors
        self.irreps_T = o3.Irreps([
            (num_features, (1, -1)),  # 1o: polar vectors
            (num_features, (1, 1)),   # 1e: pseudo vectors
            (num_features, (2, 1))    # 2e: pseudo tensors
        ])

        # Spherical harmonics irreps for edge features
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=max_l, p=-1)

        # Atomic number embedding layer
        self.embedding = Embedding(
            num_features=num_features,
            act='swish',
            max_atomic_number=max_atomic_number
        )

        # Radial basis functions for distance encoding
        self.radial_basis = Radial_Basis(
            radial_type='trainable_bessel',
            num_radial=num_radial,
            rc=cutoff,
            use_cutoff=True
        )

        # Stack of interaction blocks (message passing layers)
        self.blocks = nn.ModuleList([
            Interaction_Block(
                num_features=num_features,
                act='swish',
                head=8,
                num_radial=num_radial,
                irreps_sh=self.irreps_sh,
                irreps_T=self.irreps_T,
                dropout=0.0
            )
            for _ in range(num_blocks)
        ])

    def forward(self, z, pos, batch):
        """
        Forward pass through SE(3) backbone.

        Args:
            z: [N_total] - atomic numbers
            pos: [N_total, 3] - atomic positions
            batch: [N_total] - batch indices

        Returns:
            S: [N_total, num_features] - scalar features
            T: [N_total, irreps_T.dim] - equivariant tensor features
        """

        # Build molecular graph with radius cutoff
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=128)
        i, j = edge_index  # i: target atoms, j: source atoms

        # Compute edge distances first to filter out too-close atoms
        rij = pos[j] - pos[i]  # [num_edges, 3]
        r = torch.norm(rij, dim=-1, keepdim=True)  # [num_edges, 1]

        # Filter out edges that are too close (< 0.3 Å) to avoid numerical issues
        # This handles cases where atoms may overlap slightly in the data
        min_distance = 0.3  # Ångströms
        valid_edges = (r.squeeze(-1) >= min_distance)

        if valid_edges.sum() > 0:
            # Keep only valid edges
            edge_index = edge_index[:, valid_edges]
            i, j = edge_index
            rij = rij[valid_edges]
            r = r[valid_edges]

        # Initialize scalar features from atomic numbers
        S = self.embedding(z)  # [N_total, num_features]

        # Initialize tensor features to zeros
        T = torch.zeros(
            S.shape[0],
            self.irreps_T.dim,
            device=S.device,
            dtype=S.dtype
        )  # [N_total, irreps_T.dim]

        # Normalize edge vectors for spherical harmonics
        # Add epsilon to avoid division by zero for very close atoms
        rij_normalized = rij / (r + 1e-6)  # [num_edges, 3]

        # Compute spherical harmonics (already normalized above, so normalize=False)
        sh = o3.spherical_harmonics(
            self.irreps_sh,
            rij_normalized,
            normalize=False,
            normalization='component'
        )  # [num_edges, irreps_sh.dim]

        # Compute radial basis functions
        rbf = self.radial_basis(r.squeeze(-1))  # [num_edges, num_radial]

        # Message passing through interaction blocks
        for block in self.blocks:
            S, T = block(
                S=S,
                T=T,
                sh=sh,
                rbf=rbf,
                index=edge_index
            )

        return S, T
