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
    2. Velocity electric dipole moments (mu_vel)
    3. Magnetic dipole moments (m)
    4. Rotatory strengths (R)

    Args:
        lambda_E: Weight for excitation energy loss (default: 1.0)
        lambda_mu_vel: Weight for velocity electric dipole loss (default: 1.0)
        lambda_m: Weight for magnetic dipole loss (default: 1.0)
        lambda_R: Weight for rotatory strength loss (default: 1.0)
        lambda_R_sign: Weight for R sign classification loss (default: 1.0)
        ema_decay: EMA decay factor for loss normalization (default: 0.99)
    """

    def __init__(
        self,
        lambda_E=1.0,
        lambda_mu_vel=1.0,
        lambda_m=1.0,
        lambda_R=1.0,
        lambda_R_sign=1.0,
        ema_decay=0.99
    ):
        super().__init__()

        self.lambda_E = lambda_E
        self.lambda_mu_vel = lambda_mu_vel
        self.lambda_m = lambda_m
        self.lambda_R = lambda_R
        self.lambda_R_sign = lambda_R_sign

        # 损失归一化：用指数移动平均跟踪每个分量的量级
        self.ema_decay = ema_decay
        self.register_buffer('ema_E', torch.tensor(0.0))
        self.register_buffer('ema_mu_vel', torch.tensor(0.0))
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

        # 2. 电跃迁偶极 (Velocity Dipole) Loss - Phase-Invariant
        y_mu_vel = target['y_mu_vel'].reshape(batch_size, n_states, 3)
        mu_vel_diff = pred['mu_total_vel'] - y_mu_vel
        mu_vel_sum = pred['mu_total_vel'] + y_mu_vel
        loss_mu_vel_phase1 = (mu_vel_diff ** 2).sum(dim=-1)  # [B, n_states]
        loss_mu_vel_phase2 = (mu_vel_sum ** 2).sum(dim=-1)   # [B, n_states]
        loss_mu_vel = torch.min(loss_mu_vel_phase1, loss_mu_vel_phase2).mean()

        # 3. 磁跃迁偶极 Loss - Phase-Invariant
        y_m = target['y_m'].reshape(batch_size, n_states, 3)
        m_diff = pred['m_total'] - y_m
        m_sum = pred['m_total'] + y_m
        loss_m_phase1 = (m_diff ** 2).sum(dim=-1)  # [B, n_states]
        loss_m_phase2 = (m_sum ** 2).sum(dim=-1)   # [B, n_states]
        loss_m = torch.min(loss_m_phase1, loss_m_phase2).mean()

        # ====================================================================
        # 4. 旋转强度 R Loss = L1 + Sign Classification (BCE)
        # ====================================================================
        y_R = target['y_R'].reshape(batch_size, n_states)

        # R_pred 已经在 aggregation.py 中转换为 10^-40 cgs 单位
        # 使用 L1 loss 替代 MSE，对异常值更鲁棒，减轻过拟合
        loss_R_l1 = F.l1_loss(pred['R_pred'], y_R)

        # 符号二分类：将 R 的正负作为辅助分类任务
        y_R_sign = (y_R > 0).float()  # [B, n_states], 1=正, 0=负
        R_sign_logits = pred['R_pred']  # 直接用 R_pred 值作 logits
        loss_R_sign = F.binary_cross_entropy_with_logits(R_sign_logits, y_R_sign)

        # R 的总 loss = L1 + 分类
        loss_R = loss_R_l1

        # 计算符号预测准确率
        with torch.no_grad():
            R_sign_pred = (pred['R_pred'] > 0).float()
            r_sign_correct = (R_sign_pred == y_R_sign).float().mean()
        # ====================================================================

        # 损失归一化：用 EMA 跟踪每个分量的量级，归一化后再加权
        with torch.no_grad():
            if not self.initialized:
                self.ema_E.copy_(loss_E.detach())
                self.ema_mu_vel.copy_(loss_mu_vel.detach())
                self.ema_m.copy_(loss_m.detach())
                self.ema_R.copy_(loss_R.detach())
                self.initialized.fill_(True)
            else:
                d = self.ema_decay
                self.ema_E.mul_(d).add_(loss_E.detach(), alpha=1 - d)
                self.ema_mu_vel.mul_(d).add_(loss_mu_vel.detach(), alpha=1 - d)
                self.ema_m.mul_(d).add_(loss_m.detach(), alpha=1 - d)
                self.ema_R.mul_(d).add_(loss_R.detach(), alpha=1 - d)

        # 用 EMA 值做归一化（加 eps 防除零）
        eps = 1e-8
        norm_E = loss_E / (self.ema_E + eps)
        norm_mu_vel = loss_mu_vel / (self.ema_mu_vel + eps)
        norm_m = loss_m / (self.ema_m + eps)
        norm_R = loss_R / (self.ema_R + eps)

        # Weighted total loss
        total_loss = (
            self.lambda_E * norm_E +
            self.lambda_mu_vel * norm_mu_vel +
            self.lambda_m * norm_m +
            self.lambda_R * norm_R
        )

        loss_dict = {
            'loss': total_loss.item(),
            'loss_E': loss_E.item(),
            'loss_mu_vel': loss_mu_vel.item(),
            'loss_m': loss_m.item(),
            'loss_R': loss_R.item(),
            'loss_R_l1': loss_R_l1.item(),
            'loss_R_sign': loss_R_sign.item(),
            'R_sign_acc': r_sign_correct.item()
        }

        return total_loss, loss_dict