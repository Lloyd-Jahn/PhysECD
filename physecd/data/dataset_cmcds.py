"""
CSV 解析器和 CMCDS 数据集
=============================
本模块提供用于解析经过修正的 CMCDS_DATASET.csv 文件的实用工具。
CSV 中的所有旋转强度均已确认正确缩放至 10**-40 cgs 单位。

提取的关键物理量：
- y_E: 激发能 (eV)
- y_R: 旋转强度 [R(velocity)] (10**-40 erg-esu-cm/Gauss)
"""

import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List


class CMCDSCSVParser:
    """经过修正的 CMCDS_DATASET.csv 文件的解析器。"""

    def __init__(self, csv_path: str):
        """
        初始化 CSV 解析器。

        参数:
            csv_path: CMCDS_DATASET.csv 文件的路径
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 文件未找到: {csv_path}")

        # 读取 CSV 文件
        self.df = pd.read_csv(csv_path)
        self._validate_csv()

    def _validate_csv(self):
        """验证 CSV 文件结构。"""
        required_columns = ['ID', 'smiles', 'ECD Transition Parameters']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"缺少必需的列: {col}")

        # 检查是否恰好有 20 个激发态列
        excited_state_cols = [col for col in self.df.columns if col.startswith('Excited State_')]
        if len(excited_state_cols) != 20:
            raise ValueError(f"预期有 20 个激发态列，但找到了 {len(excited_state_cols)} 个")

    def parse_molecule(self, mol_id: int) -> Dict[str, torch.Tensor]:
        """
        解析单个分子的激发能和旋转强度。

        参数:
            mol_id: 分子 ID

        返回:
            包含以下内容的字典:
                - 'y_E': [20] 激发能 (eV)
                - 'y_R': [20] 旋转强度 (已经是 10^-40 cgs 单位)
                - 'smiles': SMILES 字符串
                - 'mol_id': 分子 ID
        """
        # 查找该分子的数据行
        mol_rows = self.df[self.df['ID'] == mol_id]

        if len(mol_rows) == 0:
            raise ValueError(f"在 CSV 中未找到分子 ID {mol_id}")

        # 转换为字典列表以便于遍历
        rows_list = mol_rows.to_dict('records')

        energy_row = None
        rotatory_row = None

        # 根据参数列识别所需的行
        for row in rows_list:
            param_name = str(row['ECD Transition Parameters'])
            if 'Excitation energies' in param_name:
                energy_row = row
            elif 'Rotatory Strength' in param_name:
                rotatory_row = row

        if energy_row is None or rotatory_row is None:
            raise ValueError(f"无法找到分子 {mol_id} 的 Excitation energies 或 Rotatory Strength 行")

        # 提取 20 个激发态列中的值
        excited_state_cols =[f'Excited State_{i}' for i in range(1, 21)]

        # 转换为 float32 张量
        y_E = torch.tensor([float(energy_row[col]) for col in excited_state_cols], dtype=torch.float32)
        y_R = torch.tensor([float(rotatory_row[col]) for col in excited_state_cols], dtype=torch.float32)

        return {
            'y_E': y_E,
            'y_R': y_R,
            'smiles': energy_row['smiles'],
            'mol_id': mol_id
        }

    def get_all_molecule_ids(self) -> List[int]:
        """
        获取 CSV 中所有唯一分子 ID 的列表。

        返回:
            分子 ID 列表
        """
        return sorted(self.df['ID'].unique().tolist())


if __name__ == "__main__":
    # 测试简化的 CSV 解析器
    csv_path = "data/CMCDS_DATASET.csv"

    try:
        parser = CMCDSCSVParser(csv_path)
        
        # 输出前几个分子的信息
        for mol_id in [1, 2, 3]:
            try:
                data = parser.parse_molecule(mol_id)
                print(f"\n分子 {mol_id}:")
                print(f"  SMILES: {data['smiles'][:50]}...")
                print(f"  激发能 [eV] (前 3 个): {data['y_E'][:3].tolist()}")
                print(f"  旋转强度 [10^-40 cgs] (前 3 个): {data['y_R'][:3].tolist()}")
            except Exception as e:
                print(f"解析分子 {mol_id} 时出错: {e}")

        mol_ids = parser.get_all_molecule_ids()
        print(f"\nCSV 中唯一的分子总数: {len(mol_ids)}")
        
    except Exception as e:
        print(f"初始化错误: {e}")