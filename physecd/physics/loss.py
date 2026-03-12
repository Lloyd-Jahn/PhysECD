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
        lambda_mu: Weight for electric dipole loss (default: 10.0)
        lambda_m: Weight for magnetic dipole loss (default: 20.0)
        lambda_R: Weight for rotatory strength loss (default: 1.0)
    """

    def __init__(
        self,
        lambda_E=1.0,
        lambda_mu=10.0,   # 提升偶极矩权重，加速底层电荷流动的学习
        lambda_m=20.0,    # 大幅提升磁偶极矩权重，打破不下降的僵局
        lambda_R=1.0      # 单位缩放对齐后，1.0 即可产生健康的梯度
    ):
        super().__init__()

        self.lambda_E = lambda_E
        self.lambda_mu = lambda_mu
        self.lambda_m = lambda_m
        self.lambda_R = lambda_R

        # ---------------- 物理常数与缩放因子 ----------------
        # 旋转强度的单位说明：
        # 模型在 aggregation.py 中已经使用公式 R = 6414.135151 × (μ · m) / E
        # 输出的 R_pred 单位已经是 10^-40 cgs，与数据集标签 y_R 单位一致
        # 因此不需要额外的单位转换

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

        # Weighted total loss
        total_loss = (
            self.lambda_E * loss_E +
            self.lambda_mu * loss_mu +
            self.lambda_m * loss_m +
            self.lambda_R * loss_R
        )

        loss_dict = {
            'loss': total_loss.item(),
            'loss_E': loss_E.item(),
            'loss_mu': loss_mu.item(),
            'loss_m': loss_m.item(),
            'loss_R': loss_R.item()
        }

        return total_loss, loss_dict

    def update_weights(self, lambda_E=None, lambda_mu=None, lambda_m=None, lambda_R=None):
        if lambda_E is not None: self.lambda_E = lambda_E
        if lambda_mu is not None: self.lambda_mu = lambda_mu
        if lambda_m is not None: self.lambda_m = lambda_m
        if lambda_R is not None: self.lambda_R = lambda_R