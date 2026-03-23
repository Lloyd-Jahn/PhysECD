"""
测试 SE3 Backbone 的旋转等变性和平移不变性。

创建分子并进行旋转和平移变换，验证模型输出的物理量是否符合等变/不变性。
"""

import torch
import numpy as np
from torch_geometric.data import Data, Batch
import sys
sys.path.append('/inspire/hdd/global_user/chenletian-240108120062/PhysECD')

from physecd.models.physecd_model import PhysECDModel


def create_test_molecule():
    """创建一个用于测试的手性分子。"""
    z = torch.tensor([6, 6, 8, 1, 1])  # C, C, O, H, H
    pos = torch.tensor([
        [0.0, 0.0, 0.0],    # C (中心)
        [1.5, 0.0, 0.0],    # C
        [2.5, 1.0, 0.0],    # O
        [0.5, 1.0, 0.5],    # H
        [-0.5, -0.5, 1.0],  # H
    ])
    return z, pos


def rotation_matrix_x(theta):
    """绕 x 轴旋转 theta 弧度。"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return torch.tensor([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t]
    ], dtype=torch.float32)


def rotation_matrix_y(theta):
    """绕 y 轴旋转 theta 弧度。"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return torch.tensor([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ], dtype=torch.float32)


def rotation_matrix_z(theta):
    """绕 z 轴旋转 theta 弧度。"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return torch.tensor([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ], dtype=torch.float32)


def apply_rotation(pos, R):
    """对坐标应用旋转矩阵。"""
    return torch.matmul(pos, R.T)


def test_rotation_equivariance():
    """测试旋转等变性。"""
    
    print("=" * 80)
    print("测试 1: 旋转等变性 (Rotation Equivariance)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建模型
    model = PhysECDModel(
        num_features=32,
        max_l=2,
        num_blocks=2,
        num_radial=16,
        cutoff=5.0,
        n_states=5,
        max_atomic_number=36
    ).to(device)
    model.eval()
    
    # 创建原始分子
    z, pos = create_test_molecule()
    z = z.to(device)
    pos = pos.to(device)
    
    # 定义多个旋转角度
    angles = [0.0, np.pi/4, np.pi/2, np.pi]
    
    print(f"\n原始分子原子坐标:")
    for i, (zi, pi) in enumerate(zip(z.cpu(), pos.cpu())):
        symbol = {1: 'H', 6: 'C', 8: 'O'}.get(int(zi), 'X')
        print(f"  {symbol}: ({pi[0]:+.3f}, {pi[1]:+.3f}, {pi[2]:+.3f})")
    
    # 存储原始结果用于对比
    batch = torch.zeros(len(z), dtype=torch.long, device=device)
    data = Data(z=z, pos=pos, batch=batch)
    
    with torch.no_grad():
        outputs_orig = model(data)
    
    mu_orig = outputs_orig['mu_total'][0].cpu()  # [5, 3]
    m_orig = outputs_orig['m_total'][0].cpu()    # [5, 3]
    E_orig = outputs_orig['E_pred'][0].cpu()     # [5]
    R_orig = outputs_orig['R_pred'][0].cpu()     # [5]
    
    print("\n" + "-" * 80)
    print("原始分子输出 (未旋转):")
    print("-" * 80)
    print_state_outputs(mu_orig, m_orig, E_orig, R_orig)
    
    # 测试不同旋转
    for angle in angles[1:]:
        print(f"\n{'=' * 80}")
        print(f"旋转角度: {angle:.4f} rad ({np.degrees(angle):.1f}°)")
        print("=" * 80)
        
        # 组合旋转 (绕 x, y, z 轴)
        R_x = rotation_matrix_x(angle).to(device)
        R_y = rotation_matrix_y(angle).to(device)
        R_z = rotation_matrix_z(angle).to(device)
        R_combined = torch.matmul(torch.matmul(R_z, R_y), R_x)
        
        # 旋转后的坐标
        pos_rotated = apply_rotation(pos, R_combined)
        
        print(f"\n旋转后分子原子坐标:")
        for i, (zi, pi) in enumerate(zip(z.cpu(), pos_rotated.cpu())):
            symbol = {1: 'H', 6: 'C', 8: 'O'}.get(int(zi), 'X')
            print(f"  {symbol}: ({pi[0]:+.3f}, {pi[1]:+.3f}, {pi[2]:+.3f})")
        
        # 前向传播
        data_rotated = Data(z=z, pos=pos_rotated, batch=batch)
        with torch.no_grad():
            outputs_rot = model(data_rotated)
        
        mu_rot = outputs_rot['mu_total'][0].cpu()  # [5, 3]
        m_rot = outputs_rot['m_total'][0].cpu()    # [5, 3]
        E_rot = outputs_rot['E_pred'][0].cpu()     # [5]
        R_rot = outputs_rot['R_pred'][0].cpu()     # [5]
        
        print(f"\n旋转后分子输出:")
        print_state_outputs(mu_rot, m_rot, E_rot, R_rot)
        
        # 验证等变性
        print(f"\n等变性验证:")
        print("-" * 80)
        
        # 电偶极矩应该随旋转矩阵变换
        mu_expected = torch.matmul(mu_orig, R_combined.cpu().T)
        mu_error = torch.norm(mu_rot - mu_expected, dim=-1).mean()
        
        # 磁偶极矩应该随旋转矩阵变换  
        m_expected = torch.matmul(m_orig, R_combined.cpu().T)
        m_error = torch.norm(m_rot - m_expected, dim=-1).mean()
        
        # 激发能量应该是旋转不变的（标量）
        E_error = torch.abs(E_rot - E_orig).mean()
        
        # 旋光强度应该是旋转不变的（标量）
        R_error = torch.abs(R_rot - R_orig).mean()
        
        print(f"电偶极矩等变误差: {mu_error:.6e}")
        print(f"磁偶极矩等变误差: {m_error:.6e}")
        print(f"激发能量不变误差: {E_error:.6e}")
        print(f"旋光强度不变误差: {R_error:.6e}")
        
        # 详细对比
        print(f"\n详细对比 (激发态 1):")
        print(f"电偶极矩:")
        print(f"  原始:    [{mu_orig[0,0]:+.6f}, {mu_orig[0,1]:+.6f}, {mu_orig[0,2]:+.6f}]")
        print(f"  旋转后:  [{mu_rot[0,0]:+.6f}, {mu_rot[0,1]:+.6f}, {mu_rot[0,2]:+.6f}]")
        print(f"  期望:    [{mu_expected[0,0]:+.6f}, {mu_expected[0,1]:+.6f}, {mu_expected[0,2]:+.6f}]")
        print(f"磁偶极矩:")
        print(f"  原始:    [{m_orig[0,0]:+.6f}, {m_orig[0,1]:+.6f}, {m_orig[0,2]:+.6f}]")
        print(f"  旋转后:  [{m_rot[0,0]:+.6f}, {m_rot[0,1]:+.6f}, {m_rot[0,2]:+.6f}]")
        print(f"  期望:    [{m_expected[0,0]:+.6f}, {m_expected[0,1]:+.6f}, {m_expected[0,2]:+.6f}]")


def test_translation_invariance():
    """测试平移不变性。"""
    
    print("\n" + "=" * 80)
    print("测试 2: 平移不变性 (Translation Invariance)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = PhysECDModel(
        num_features=32,
        max_l=2,
        num_blocks=2,
        num_radial=16,
        cutoff=5.0,
        n_states=5,
        max_atomic_number=36
    ).to(device)
    model.eval()
    
    # 创建原始分子
    z, pos = create_test_molecule()
    z = z.to(device)
    pos = pos.to(device)
    
    # 定义不同的平移向量
    translations = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [10.0, -5.0, 2.5],
    ]
    
    print(f"\n原始分子质心: {pos.mean(dim=0).cpu().numpy()}")
    
    batch = torch.zeros(len(z), dtype=torch.long, device=device)
    data = Data(z=z, pos=pos, batch=batch)
    
    with torch.no_grad():
        outputs_orig = model(data)
    
    mu_orig = outputs_orig['mu_total'][0].cpu()
    m_orig = outputs_orig['m_total'][0].cpu()
    E_orig = outputs_orig['E_pred'][0].cpu()
    R_orig = outputs_orig['R_pred'][0].cpu()
    
    print("\n原始分子输出:")
    print_state_outputs(mu_orig, m_orig, E_orig, R_orig)
    
    for trans in translations[1:]:
        trans_vec = torch.tensor(trans, device=device, dtype=torch.float32)
        
        print(f"\n{'=' * 80}")
        print(f"平移向量: [{trans[0]:+.1f}, {trans[1]:+.1f}, {trans[2]:+.1f}]")
        print("=" * 80)
        
        # 平移后的坐标
        pos_translated = pos + trans_vec.unsqueeze(0)
        
        print(f"平移后分子质心: {pos_translated.mean(dim=0).cpu().numpy()}")
        
        # 前向传播
        data_translated = Data(z=z, pos=pos_translated, batch=batch)
        with torch.no_grad():
            outputs_trans = model(data_translated)
        
        mu_trans = outputs_trans['mu_total'][0].cpu()
        m_trans = outputs_trans['m_total'][0].cpu()
        E_trans = outputs_trans['E_pred'][0].cpu()
        R_trans = outputs_trans['R_pred'][0].cpu()
        
        print(f"\n平移后分子输出:")
        print_state_outputs(mu_trans, m_trans, E_trans, R_trans)
        
        # 验证不变性
        print(f"\n不变性验证:")
        print("-" * 80)
        
        mu_error = torch.norm(mu_trans - mu_orig, dim=-1).mean()
        m_error = torch.norm(m_trans - m_orig, dim=-1).mean()
        E_error = torch.abs(E_trans - E_orig).mean()
        R_error = torch.abs(R_trans - R_orig).mean()
        
        print(f"电偶极矩不变误差: {mu_error:.6e}")
        print(f"磁偶极矩不变误差: {m_error:.6e}")
        print(f"激发能量不变误差: {E_error:.6e}")
        print(f"旋光强度不变误差: {R_error:.6e}")


def print_state_outputs(mu, m, E, R, max_states=3):
    """打印前几个激发态的输出。"""
    print("-" * 80)
    for state in range(min(len(E), max_states)):
        print(f"\n激发态 {state + 1}:")
        print(f"  E = {E[state]:.6f} eV")
        print(f"  μ = [{mu[state,0]:+.6f}, {mu[state,1]:+.6f}, {mu[state,2]:+.6f}] (e·Å)")
        print(f"  m = [{m[state,0]:+.6f}, {m[state,1]:+.6f}, {m[state,2]:+.6f}] (μB)")
        print(f"  R = {R[state]:.6f} (10^-40 cgs)")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SE(3) 等变性/不变性测试")
    print("=" * 80)
    print("\n本测试验证:")
    print("1. 旋转等变性: 输入旋转 → 向量输出相应旋转")
    print("2. 平移不变性: 输入平移 → 所有输出保持不变")
    print("\n期望值:")
    print("- 电偶极矩 μ: 旋转等变, 平移不变")
    print("- 磁偶极矩 m: 旋转等变, 平移不变")
    print("- 激发能量 E: 旋转不变, 平移不变")
    print("- 旋光强度 R: 旋转不变, 平移不变")
    
    test_rotation_equivariance()
    test_translation_invariance()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
