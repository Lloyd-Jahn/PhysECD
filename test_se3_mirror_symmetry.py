"""
测试 SE3 Backbone 的镜面对称性。

创建两个沿 yz 平面镜面对称的分子，测试模型输出的物理量是否符合对称性。
"""

import torch
from torch_geometric.data import Data, Batch
import sys
sys.path.append('/inspire/hdd/global_user/chenletian-240108120062/PhysECD')

from physecd.models.physecd_model import PhysECDModel


def create_mirror_pair():
    """
    创建两个沿 yz 平面镜面对称的简单分子。
    
    沿 yz 平面镜像变换: (x, y, z) -> (-x, y, z)
    """
    # 创建第一个分子 (左手性)
    # 简单的 4 原子分子，形成一个手性结构
    z1 = torch.tensor([6, 6, 8, 1])  # C, C, O, H
    pos1 = torch.tensor([
        [0.0, 0.0, 0.0],   # C (中心)
        [1.5, 0.0, 0.0],   # C
        [2.5, 1.0, 0.0],   # O
        [0.5, 1.0, 0.5],   # H (产生手性)
    ])
    batch1 = torch.zeros(4, dtype=torch.long)
    
    # 创建第二个分子 (右手性，沿 yz 平面的镜像)
    # 镜像变换: x -> -x
    z2 = z1.clone()
    pos2 = pos1.clone()
    pos2[:, 0] = -pos2[:, 0]  # x 坐标取反
    batch2 = torch.ones(4, dtype=torch.long)
    
    return z1, pos1, batch1, z2, pos2, batch2


def test_se3_backbone():
    """测试 SE3 Backbone 的对称性输出。"""
    
    print("=" * 80)
    print("测试 SE3 Backbone: 沿 yz 平面镜面对称的分子对")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建模型
    model = PhysECDModel(
        num_features=32,      # 使用较小的特征数便于测试
        max_l=2,
        num_blocks=2,
        num_radial=16,
        cutoff=5.0,
        n_states=5,           # 测试 5 个激发态
        max_atomic_number=36
    ).to(device)
    
    model.eval()  # 评估模式
    
    # 创建镜面对称的分子对
    z1, pos1, batch1, z2, pos2, batch2 = create_mirror_pair()
    
    # 移动到设备
    z1 = z1.to(device)
    pos1 = pos1.to(device)
    batch1 = batch1.to(device)
    z2 = z2.to(device)
    pos2 = pos2.to(device)
    batch2 = batch2.to(device)
    
    print("\n分子 1 (原始) 原子坐标:")
    for i, (z, p) in enumerate(zip(z1.cpu(), pos1.cpu())):
        symbol = {1: 'H', 6: 'C', 8: 'O'}.get(int(z), 'X')
        print(f"  {symbol}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
    
    print("\n分子 2 (镜像) 原子坐标:")
    for i, (z, p) in enumerate(zip(z2.cpu(), pos2.cpu())):
        symbol = {1: 'H', 6: 'C', 8: 'O'}.get(int(z), 'X')
        print(f"  {symbol}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
    
    # 创建 batch
    data1 = Data(z=z1, pos=pos1, batch=batch1)
    data2 = Data(z=z2, pos=pos2, batch=batch2)
    batch_data = Batch.from_data_list([data1, data2])
    
    # 前向传播
    with torch.no_grad():
        outputs = model(batch_data)
    
    # 提取结果
    E_pred = outputs['E_pred']      # [2, 5] 激发能量
    mu_total = outputs['mu_total']  # [2, 5, 3] 跃迁电偶极矩
    m_total = outputs['m_total']    # [2, 5, 3] 跃迁磁偶极矩
    R_pred = outputs['R_pred']      # [2, 5] 旋光强度
    
    # 打印详细结果
    print("\n" + "=" * 80)
    print("模型输出")
    print("=" * 80)
    
    for state in range(5):
        print(f"\n--- 激发态 {state + 1} ---")
        
        # 分子 1
        E1 = E_pred[0, state].item()
        mu1 = mu_total[0, state].cpu().numpy()
        m1 = m_total[0, state].cpu().numpy()
        R1 = R_pred[0, state].item()
        
        # 分子 2
        E2 = E_pred[1, state].item()
        mu2 = mu_total[1, state].cpu().numpy()
        m2 = m_total[1, state].cpu().numpy()
        R2 = R_pred[1, state].item()
        
        print(f"\n分子 1:")
        print(f"  激发能量 E = {E1:.6f} eV")
        print(f"  跃迁电偶极矩 μ = [{mu1[0]:+.6f}, {mu1[1]:+.6f}, {mu1[2]:+.6f}] (e·Å)")
        print(f"  跃迁磁偶极矩 m = [{m1[0]:+.6f}, {m1[1]:+.6f}, {m1[2]:+.6f}] (μB)")
        print(f"  旋光强度 R = {R1:.6f} (10^-40 cgs)")
        
        print(f"\n分子 2 (镜像):")
        print(f"  激发能量 E = {E2:.6f} eV")
        print(f"  跃迁电偶极矩 μ = [{mu2[0]:+.6f}, {mu2[1]:+.6f}, {mu2[2]:+.6f}] (e·Å)")
        print(f"  跃迁磁偶极矩 m = [{m2[0]:+.6f}, {m2[1]:+.6f}, {m2[2]:+.6f}] (μB)")
        print(f"  旋光强度 R = {R2:.6f} (10^-40 cgs)")
        
        # 分析对称性
        print(f"\n对称性分析:")
        print(f"  能量差 |E1 - E2| = {abs(E1 - E2):.6e} eV")
        print(f"  电偶极矩 x 分量: μ1_x = {mu1[0]:+.6f}, μ2_x = {mu2[0]:+.6f}")
        print(f"  电偶极矩 y 分量: μ1_y = {mu1[1]:+.6f}, μ2_y = {mu2[1]:+.6f}")
        print(f"  电偶极矩 z 分量: μ1_z = {mu1[2]:+.6f}, μ2_z = {mu2[2]:+.6f}")
        print(f"  磁偶极矩 x 分量: m1_x = {m1[0]:+.6f}, m2_x = {m2[0]:+.6f}")
        print(f"  磁偶极矩 y 分量: m1_y = {m1[1]:+.6f}, m2_y = {m2[1]:+.6f}")
        print(f"  磁偶极矩 z 分量: m1_z = {m1[2]:+.6f}, m2_z = {m2[2]:+.6f}")
        print(f"  旋光强度 R1 = {R1:+.6f}, R2 = {R2:+.6f}")
        print(f"  旋光强度比 R2/R1 = {R2/R1 if abs(R1) > 1e-10 else 'N/A':.6f}")
    
    # 总结期望的对称性行为
    print("\n" + "=" * 80)
    print("期望的对称性行为 (yz 平面镜像):")
    print("=" * 80)
    print("- 电偶极矩 (μ): x 分量变号, y 和 z 分量不变")
    print("  即: μ2 = (-μ1_x, μ1_y, μ1_z)")
    print("- 磁偶极矩 (m): x 分量不变, y 和 z 分量变号")
    print("  即: m2 = (m1_x, -m1_y, -m1_z)")
    print("- 旋光强度 (R): 应该变号 (R2 = -R1)")
    print("  因为 R ∝ μ · m")
    print("- 激发能量 (E): 应该相同 (标量)")
    
    # 验证旋光强度的反对称性
    print("\n" + "=" * 80)
    print("旋光强度反对称性验证:")
    print("=" * 80)
    for state in range(5):
        R1 = R_pred[0, state].item()
        R2 = R_pred[1, state].item()
        sum_R = R1 + R2
        print(f"态 {state+1}: R1 = {R1:+.6f}, R2 = {R2:+.6f}, R1+R2 = {sum_R:+.6e}")


if __name__ == "__main__":
    test_se3_backbone()
