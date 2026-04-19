import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PhysECDLoss(nn.Module):
    """
    Differentiable Spectrum Broadening Loss
    计算基于高斯展宽的宏观光谱 Loss，消除能级交叉带来的微观排序惩罚。
    """
    def __init__(self, sigma=0.4, wl_min=80.0, wl_max=450.0, wl_step=1.0):
        super().__init__()
        self.sigma = sigma
        
        # 1. 预先构建波长网格 (Wavelength grid)
        # 与 05_generate_pred_spectrum.py 保持绝对一致
        wl_grid = torch.arange(wl_min, wl_max + wl_step, wl_step)
        
        # 2. 转换为能量网格 (Energy grid, eV)
        e_grid = 1240.0 / wl_grid
        
        # 3. 注册为 buffer，确保它能随着模型自动移动到 GPU/CPU
        # 维度设为 [1, 1, N_points] 以便后续直接进行张量广播
        self.register_buffer('e_grid', e_grid.view(1, 1, -1))
        
        # 4. 物理常数
        self.norm_constant = 2.296e1 * sigma * math.sqrt(math.pi)
        self.theta_factor = 3298.2

    def generate_differentiable_spectrum(self, E_states, R_cgs):
        """
        核心可微操作：将 20 个离散的 (E, R) 转化为连续的光谱曲线
        E_states, R_cgs shape:[Batch_size, 20]
        """
        # 扩展维度以进行广播: [B, 20] ->[B, 20, 1]
        E_ext = E_states.unsqueeze(2)
        R_ext = R_cgs.unsqueeze(2)
        
        # 计算高斯核: exp(-((E_grid - E_i) / sigma)^2)
        # e_grid[1, 1, N_points] - E_ext [B, 20, 1] ->[B, 20, N_points]
        gaussian = torch.exp(-((self.e_grid - E_ext) / self.sigma) ** 2)
        
        # 严格执行物理公式: Δε = Σ E_i * R_i * gaussian
        # sum(dim=1) 把 20 个态的贡献叠加
        delta_eps = torch.sum(E_ext * R_ext * gaussian, dim=1) / self.norm_constant  # [B, N_points]
        
        # 转化为摩尔椭圆度 [θ]
        theta = delta_eps * self.theta_factor  # [B, N_points]
        
        return theta

    def forward(self, output_dict, batch_data):
        """
        计算各种 Loss 组件
        """
        # 模型预测值
        E_pred = output_dict['E_pred']  #[B, 20]
        R_pred = output_dict['R_pred']  #[B, 20]
        
        # 获取 Batch Size 和状态数
        batch_size = E_pred.shape[0]
        n_states = E_pred.shape[1]
        
        # 真实标签值
        E_target = batch_data.y_E.view(batch_size, n_states)
        R_target = batch_data.y_R.view(batch_size, n_states)
        
        # ---------------------------------------------------------
        # 1. 生成宏观连续光谱
        # ---------------------------------------------------------
        pred_spectrum = self.generate_differentiable_spectrum(E_pred, R_pred)
        target_spectrum = self.generate_differentiable_spectrum(E_target, R_target)
        
        # ---------------------------------------------------------
        # 2. 形状匹配 Loss (最核心，关注峰的正负和相对位置)
        # ---------------------------------------------------------
        # 减去均值 (Centered Cosine Similarity 等价于 Pearson Correlation)
        pred_centered = pred_spectrum - pred_spectrum.mean(dim=1, keepdim=True)
        target_centered = target_spectrum - target_spectrum.mean(dim=1, keepdim=True)
        
        # cos_sim 范围 [-1, 1], 越接近 1 越好
        cos_sim = F.cosine_similarity(pred_centered, target_centered, dim=1, eps=1e-8)
        loss_shape = (1.0 - cos_sim).mean()  # 范围 [0, 2]
        
        # ---------------------------------------------------------
        # 3. 幅度匹配 Loss (限制预测出的绝对方差，避免极值)
        # ---------------------------------------------------------
        # 因为 [θ] 的值域非常大 (10^4 ~ 10^5)，直接做 L1 会导致 Loss 爆炸
        # 我们在这里统一乘以 1e-4 进行缩放计算
        scale = 1e-4
        loss_mag = F.smooth_l1_loss(pred_spectrum * scale, target_spectrum * scale)
        
        # ---------------------------------------------------------
        # 4. 辅助：能量锚定 Loss
        # ---------------------------------------------------------
        # E 的排序交叉相对较少，提供一点微观监督可以帮助 SE(3) Backbone 加速收敛
        loss_E = F.mse_loss(E_pred, E_target)
        
        # ---------------------------------------------------------
        # 5. 辅助：微观参数正则化 (消除内部神仙打架)
        # ---------------------------------------------------------
        loss_reg = torch.tensor(0.0, device=E_pred.device)
        if 'mu_pred' in output_dict and 'm_pred' in output_dict:
            # 迫使模型寻找最平滑、最经济的原子级偶极分布
            loss_reg = torch.mean(output_dict['mu_pred']**2) + torch.mean(output_dict['m_pred']**2)
            
        return {
            'loss_shape': loss_shape,
            'loss_mag': loss_mag,
            'loss_E': loss_E,
            'loss_reg': loss_reg
        }