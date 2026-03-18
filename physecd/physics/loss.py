"""
Multi-task loss function for PhysECD training.

Implements weighted MSE loss across multiple physical quantities,
with strict consideration of Quantum Chemistry unit conversions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysECDLoss(nn.Module):
    """
    Multi-task MSE loss for PhysECD model.

    Computes weighted mean squared error across 4 physical quantities:
    1. Excitation energies (E)
    2. Electric dipole moments (mu)
    3. Magnetic dipole moments (m)
    4. Rotatory strengths (R)

    Args:
        lambda_E: Weight for excitation energy loss (default: 1.0)
        lambda_mu: Weight for electric dipole loss (default: 1.0)
        lambda_m: Weight for magnetic dipole loss (default: 1.0)
        lambda_R: Weight for rotatory strength loss (default: 1.0)
        ema_decay: EMA decay factor for loss normalization (default: 0.99)
    """

    def __init__(
        self,
        lambda_E=1.0,
        lambda_mu=1.0,
        lambda_m=1.0,
        lambda_R=1.0,
        ema_decay=0.99
    ):
        super().__init__()

        self.lambda_E = lambda_E
        self.lambda_mu = lambda_mu
        self.lambda_m = lambda_m
        self.lambda_R = lambda_R

        # 损失归一化：用指数移动平均跟踪每个分量的量级
        self.ema_decay = ema_decay
        self.register_buffer('ema_E', torch.tensor(0.0))
        self.register_buffer('ema_mu', torch.tensor(0.0))
        self.register_buffer('ema_m', torch.tensor(0.0))
        self.register_buffer('ema_R', torch.tensor(0.0))
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, pred, target):
        """
        Compute multi-task loss.

        Args:
            pred: Dictionary of predictions from model
                - E_pred:[Batch_size, 20]
                - mu_total: [Batch_size, 20, 3] (in a.u.)
                - m_total: [Batch_size, 20, 3] (in a.u.)
                - R_pred: [Batch_size, 20] (in a.u., calculated via dot product)

            target: Dictionary of ground truth values
                - y_E:[Batch_size, 20]
                - y_mu_vel:[Batch_size, 20, 3] (in a.u.)
                - y_m:[Batch_size, 20, 3] (in a.u.)
                - y_R: [Batch_size, 20] (in 10^-40 cgs)

        Returns:
            total_loss: Scalar tensor with weighted total loss
            loss_dict: Dictionary with loss values for logging
        """
        # PyG DataLoader 将标签直接拼接，需要 reshape 对齐预测值的形状
        batch_size = pred['E_pred'].shape[0]
        n_states   = pred['E_pred'].shape[1]

        # 1. 激发能 Loss
        y_E = target['y_E'].reshape(batch_size, n_states)
        loss_E = F.mse_loss(pred['E_pred'], y_E)

        # 2. 电跃迁偶极 (Velocity Dipole) Loss
        y_mu = target['y_mu_vel'].reshape(batch_size, n_states, 3)
        loss_mu = F.mse_loss(pred['mu_total'], y_mu)

        # 3. 磁跃迁偶极 Loss
        y_m = target['y_m'].reshape(batch_size, n_states, 3)
        loss_m = F.mse_loss(pred['m_total'], y_m)

        # ====================================================================
        # 4. 旋转强度 R Loss
        # ====================================================================
        y_R = target['y_R'].reshape(batch_size, n_states)

        # R_pred 已经在 aggregation.py 中转换为 10^-40 cgs 单位
        # 直接与 y_R 比较即可
        loss_R = F.mse_loss(pred['R_pred'], y_R)
        # ====================================================================

        # 损失归一化：用 EMA 跟踪每个分量的量级，归一化后再加权
        with torch.no_grad():
            if not self.initialized:
                self.ema_E.copy_(loss_E.detach())
                self.ema_mu.copy_(loss_mu.detach())
                self.ema_m.copy_(loss_m.detach())
                self.ema_R.copy_(loss_R.detach())
                self.initialized.fill_(True)
            else:
                d = self.ema_decay
                self.ema_E.mul_(d).add_(loss_E.detach(), alpha=1 - d)
                self.ema_mu.mul_(d).add_(loss_mu.detach(), alpha=1 - d)
                self.ema_m.mul_(d).add_(loss_m.detach(), alpha=1 - d)
                self.ema_R.mul_(d).add_(loss_R.detach(), alpha=1 - d)

        # 用 EMA 值做归一化（加 eps 防除零）
        eps = 1e-8
        norm_E = loss_E / (self.ema_E + eps)
        norm_mu = loss_mu / (self.ema_mu + eps)
        norm_m = loss_m / (self.ema_m + eps)
        norm_R = loss_R / (self.ema_R + eps)

        # Weighted total loss
        total_loss = (
            self.lambda_E * norm_E +
            self.lambda_mu * norm_mu +
            self.lambda_m * norm_m +
            self.lambda_R * norm_R
        )

        loss_dict = {
            'loss': total_loss.item(),
            'loss_E': loss_E.item(),
            'loss_mu': loss_mu.item(),
            'loss_m': loss_m.item(),
            'loss_R': loss_R.item()
        }

        return total_loss, loss_dict