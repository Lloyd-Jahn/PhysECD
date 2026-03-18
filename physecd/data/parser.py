"""
Gaussian Log 解析器
===================
本模块提供了一个解析器，用于完全从 Gaussian TD-DFT 计算输出文件 (.log) 中提取物理量，以确保物理一致性。

提取的关键物理量：
1. 来自标准方向 (Standard orientation) 的原子坐标 (pos) 和原子序数 (z)
2. 速度电偶极矩 (Velocity electric dipole moments, mu_vel)
3. 磁偶极矩 (Magnetic dipole moments, m)
"""

import torch
from pathlib import Path
from typing import Dict, Tuple


class GaussianParser:
    """Gaussian TD-DFT 计算 .log 文件的解析器。"""

    def __init__(self, log_dir: str):
        """
        使用 log 目录路径初始化解析器。

        参数:
            log_dir: 包含 .log 文件的目录路径 (例如 6.ECD_LOG)
        """
        self.log_dir = Path(log_dir)

        if not self.log_dir.exists():
            raise FileNotFoundError(f"未找到 LOG 目录: {log_dir}")

    def parse_molecule(self, mol_id: int, n_states: int = 20) -> Dict[str, torch.Tensor]:
        """
        解析 .log 文件以提取分子的所有所需属性。

        参数:
            mol_id: 分子 ID (例如，1 对应 molecule_1_ECD.log)
            n_states: 激发态数量 (默认: 20)

        返回:
            包含以下内容的字典:
                - 'pos': [N_atoms, 3] 原子坐标 (Standard orientation)
                - 'z': [N_atoms] 原子序数
                - 'y_mu_vel': [n_states, 3] 速度偶极矩
                - 'y_m': [n_states, 3] 磁偶极矩
        """
        log_path = self.log_dir / f"molecule_{mol_id}_ECD.log"

        if not log_path.exists():
            raise FileNotFoundError(f"未找到 LOG 文件: {log_path}")

        # 逐行读取整个 log 文件
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 从这些行中提取所有物理量
        pos, z = self._extract_coordinates(lines)
        mu_vel = self._extract_velocity_dipole(lines, n_states)
        m = self._extract_magnetic_dipole(lines, n_states)

        return {
            'pos': pos,
            'z': z,
            'y_mu_vel': mu_vel,
            'y_m': m,
            'mol_id': mol_id
        }

    def _extract_coordinates(self, lines: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 'Standard orientation' 块中提取坐标和原子序数。
        始终从下往上搜索，以获取最终收敛的几何结构 (如果发生了优化)。
        """
        start_idx = -1
        # 反向搜索以找到最后一个 Standard orientation 块
        for i in range(len(lines) - 1, -1, -1):
            if "Standard orientation:" in lines[i]:
                start_idx = i
                break

        if start_idx == -1:
            raise ValueError("在 log 文件中找不到 'Standard orientation:'")

        # 跳过 5 行标题:
        # 1. ---------------------------------------------------------------------
        # 2. Center     Atomic      Atomic             Coordinates (Angstroms)
        # 3. Number     Number       Type             X           Y           Z
        # 4. ---------------------------------------------------------------------
        current_idx = start_idx + 5
        
        atomic_numbers = []
        coordinates =[]

        while current_idx < len(lines):
            line = lines[current_idx].strip()
            # 停止条件: 遇到下一个破折号分隔符
            if line.startswith('---'):
                break

            parts = line.split()
            if len(parts) == 6:
                # 格式: [中心编号, 原子序数, 原子类型, X, Y, Z]
                atomic_numbers.append(int(parts[1]))
                coordinates.append([float(parts[3]), float(parts[4]), float(parts[5])])
            current_idx += 1

        if not atomic_numbers:
            raise ValueError("在 Standard orientation 块中未找到原子坐标。")

        pos = torch.tensor(coordinates, dtype=torch.float32)
        z = torch.tensor(atomic_numbers, dtype=torch.long)

        return pos, z

    def _extract_velocity_dipole(self, lines: list, n_states: int) -> torch.Tensor:
        """
        提取速度电偶极矩。
        """
        start_idx = -1
        for i, line in enumerate(lines):
            if "Ground to excited state transition velocity dipole moments (Au):" in line:
                start_idx = i
                break
                
        if start_idx == -1:
            raise ValueError("在 log 文件中找不到速度偶极矩部分")

        # 跳过标题行 (1. 标题, 2. 列标题)
        current_idx = start_idx + 2
        dipoles =[]
        
        while current_idx < len(lines) and len(dipoles) < n_states:
            line = lines[current_idx].strip()
            parts = line.split()
            
            # 行格式: 状态(int) X(float) Y(float) Z(float) Dip.S. Osc.
            if len(parts) >= 4 and parts[0].isdigit():
                dipoles.append([float(parts[1]), float(parts[2]), float(parts[3])])
            current_idx += 1

        if len(dipoles) != n_states:
            raise ValueError(f"预期 {n_states} 个状态，但在速度偶极矩部分找到了 {len(dipoles)} 个")

        return torch.tensor(dipoles, dtype=torch.float32)

    def _extract_magnetic_dipole(self, lines: list, n_states: int) -> torch.Tensor:
        """
        提取磁偶极矩。
        """
        start_idx = -1
        for i, line in enumerate(lines):
            if "Ground to excited state transition magnetic dipole moments (Au):" in line:
                start_idx = i
                break
                
        if start_idx == -1:
            raise ValueError("在 log 文件中找不到磁偶极矩部分")

        # 跳过标题行 (1. 标题, 2. 列标题)
        current_idx = start_idx + 2
        dipoles =[]
        
        while current_idx < len(lines) and len(dipoles) < n_states:
            line = lines[current_idx].strip()
            parts = line.split()
            
            # 行格式: 状态(int) X(float) Y(float) Z(float)
            if len(parts) >= 4 and parts[0].isdigit():
                dipoles.append([float(parts[1]), float(parts[2]), float(parts[3])])
            current_idx += 1

        if len(dipoles) != n_states:
            raise ValueError(f"预期 {n_states} 个状态，但在磁偶极矩部分找到了 {len(dipoles)} 个")

        return torch.tensor(dipoles, dtype=torch.float32)


if __name__ == "__main__":
    # 测试 .log 解析器
    log_dir = "/Users/jiangyi/Desktop/ECD光谱预测大模型课题文件/PhysECD/Raw_data_gaussian_input_output_V1/Raw_data_gaussian_input_output/Raw_data_gaussian_input_output/Raw_Data_ECD_gaussian_input_output/6.ECD_LOG"

    parser = GaussianParser(log_dir)

    # 在分子 1 上进行测试
    try:
        data = parser.parse_molecule(1)
        print("成功解析分子 1:")
        print(f"  - 原子数量: {len(data['z'])}")
        print(f"  - 坐标形状: {data['pos'].shape}")
        print(f"  - 速度偶极矩形状: {data['y_mu_vel'].shape}")
        print(f"  - 磁偶极矩形状: {data['y_m'].shape}")
        print(f"\n前 3 个原子:")
        for i in range(3):
            print(f"  原子 {i+1}: Z={data['z'][i].item()}, pos={data['pos'][i].numpy()}")
        print(f"\n前 3 个激发态 (速度偶极矩):")
        print(data['y_mu_vel'][:3])
        print(f"\n前 3 个激发态 (磁偶极矩):")
        print(data['y_m'][:3])
    except Exception as e:
        print(f"错误: {e}")