"""
生成 CMCDS 数据集的对映异构体数据
===========================================
本脚本通过应用几何和矢量的对称性变换，为数据集中的所有分子生成对映异构体（镜像）数据。

变换规则：跨 XY 平面进行镜像反射
- 空间坐标：(x, y, z) → (x, y, -z)
- 电偶极矩（极矢量 polar vector）：(μx, μy, μz) → (μx, μy, -μz)
- 磁偶极矩（轴矢量 axial vector）：(mx, my, mz) → (-mx, -my, mz)
- 旋转强度：R → -R
- 激发能：保持不变
- SMILES：手性中心标识 '@' 和 '@@' 互换

此操作在不需要进行新的 DFT 计算的情况下，从物理底层将数据集的规模扩大了一倍 (10887 -> 21774)。
"""

import sys
from pathlib import Path
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import argparse
import pandas as pd

# 将父目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='为 CMCDS 数据集生成对映异构体数据')

    parser.add_argument(
        '--input_dir',
        type=str,
        default='/home/data/jiangyi/PhysECD-3.17修改-公式严谨性检查+单独训练R/data/processed',
        help='包含原始 .pt 数据文件的目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/data/jiangyi/PhysECD-3.17修改-公式严谨性检查+单独训练R/data/processed_with_enantiomers',
        help='包含扩展后 .pt 文件的输出目录'
    )
    parser.add_argument(
        '--csv_output',
        type=str,
        default='/home/data/jiangyi/PhysECD-3.17修改-公式严谨性检查+单独训练R/data/CMCDS_DATASET_with_enantiomers.csv',
        help='扩展后的 CSV 文件的输出路径'
    )
    parser.add_argument(
        '--original_csv',
        type=str,
        default='/home/data/jiangyi/PhysECD-3.17修改-公式严谨性检查+单独训练R/data/CMCDS_DATASET.csv',
        help='CMCDS_DATASET.csv 路径（用于提取波长数据）'
    )

    return parser.parse_args()


def generate_enantiomer(data: Data) -> Data:
    """
    通过跨 XY 平面的反射生成对映异构体数据。

    科学原理与数学对称性验证：
    1. 坐标 (x, y, z) → (x, y, -z)
    2. 电偶极矩（极矢量）: (μx, μy, μz) → (μx, μy, -μz)
    3. 磁偶极矩（伪矢量/轴矢量）: (mx, my, mz) → (-mx, -my, mz)
    4. 旋转强度: R ∝ Im(μ·m) → -R
    5. 激发能: 标量，保持不变
    6. SMILES: 手性中心标识 '@' 和 '@@' 互换

    参数:
        data: 原始的 PyG Data 对象
    返回:
        对映异构体的 PyG Data 对象
    """
    enantiomer = Data()

    # 1. 坐标变换：反转 Z 轴
    pos_enantiomer = data.pos.clone()
    pos_enantiomer[:, 2] = -pos_enantiomer[:, 2]

    # 2. 速度电偶极矩（极矢量）变换：反转 Z 分量
    y_mu_vel_enantiomer = data.y_mu_vel.clone()
    y_mu_vel_enantiomer[:, 2] = -y_mu_vel_enantiomer[:, 2]

    # 3. 磁偶极矩（轴矢量）变换：反转 X 和 Y 分量
    y_m_enantiomer = data.y_m.clone()
    y_m_enantiomer[:, 0] = -y_m_enantiomer[:, 0]
    y_m_enantiomer[:, 1] = -y_m_enantiomer[:, 1]

    # 4. 旋转强度变换：取相反数
    y_R_enantiomer = -data.y_R.clone()

    # 5. 激发能与原子序数保持不变
    y_E_enantiomer = data.y_E.clone()
    z_enantiomer = data.z.clone()

    # 6. 翻转 SMILES 中的手性中心标识
    # 将 '@@' 替换为临时字符串符，避免被下一步的 '@' 替换逻辑覆盖，最后再互换。
    temp_smiles = data.smiles.replace('@@', '__TEMP__')
    temp_smiles = temp_smiles.replace('@', '@@')
    
    # 组装对映异构体 Data 对象
    enantiomer.z = z_enantiomer
    enantiomer.pos = pos_enantiomer
    enantiomer.y_E = y_E_enantiomer
    enantiomer.y_mu_vel = y_mu_vel_enantiomer
    enantiomer.y_m = y_m_enantiomer
    enantiomer.y_R = y_R_enantiomer
    enantiomer.smiles = temp_smiles.replace('__TEMP__', '@')
    enantiomer.mol_id = -data.mol_id  # 使用负数的 ID 标识对映异构体

    return enantiomer


def verify_transformation(original: Data, enantiomer: Data):
    """
    通过检验 μ·m 的内积符号变化来验证对称性变换是否正确。
    """
    mu_dot_m_original = (original.y_mu_vel * original.y_m).sum(dim=1)
    mu_dot_m_enantiomer = (enantiomer.y_mu_vel * enantiomer.y_m).sum(dim=1)

    # 计算绝对误差，避免除以负数时引发的符号漂移问题
    max_error = torch.abs(mu_dot_m_enantiomer + mu_dot_m_original).max().item()
    return max_error


def expand_dataset(data_list, verify=True):
    """扩展数据集（为所有分子生成对映异构体）。"""
    expanded_data = []
    max_errors =[]

    for data in tqdm(data_list, desc="  生成对映异构体"):
        expanded_data.append(data)
        enantiomer = generate_enantiomer(data)
        expanded_data.append(enantiomer)

        if verify:
            error = verify_transformation(data, enantiomer)
            max_errors.append(error)

    if verify and len(max_errors) > 0:
        avg_error = sum(max_errors) / len(max_errors)
        max_error = max(max_errors)
        print(f"  验证情况: 平均误差 = {avg_error:.2e}, 最大误差 = {max_error:.2e}")
        if max_error > 0.01:
            print(f"  警告: 发现较大的变换误差，请检查物理量！")

    return expanded_data


def load_wavelengths_from_csv(csv_path):
    """
    从 CMCDS_DATASET.csv 中加载波长数据。
    """
    df = pd.read_csv(csv_path)
    excited_state_cols =[f'Excited State_{i}' for i in range(1, 21)]
    wavelength_map = {}

    wl_rows = df[df['ECD Transition Parameters'].str.contains('Wavelengths', na=False)]
    
    for _, row in wl_rows.iterrows():
        mol_id = int(row['ID'])
        wavelength_map[mol_id] = [row[col] for col in excited_state_cols]

    return wavelength_map


def generate_expanded_csv(train_data, val_data, test_data, output_path, wavelength_map):
    """
    生成包含原分子和对映异构体数据的扩展版 CSV 文件。
    """
    all_data = train_data + val_data + test_data
    # 按照绝对值升序排序，同一分子的正负 ID 靠在一起
    all_data = sorted(all_data, key=lambda x: (abs(x.mol_id), -x.mol_id))

    rows =[]
    missing_wavelengths = 0
    
    for data in tqdm(all_data, desc="  构建 CSV 行数据"):
        mol_id = data.mol_id
        smiles = data.smiles
        original_mol_id = abs(mol_id)

        # 1. 激发能行
        energy_row = {
            'ID': mol_id,
            'smiles': smiles,
            'ECD Transition Parameters': 'Excitation energies (eV)'
        }
        for i, energy in enumerate(data.y_E.tolist(), 1):
            energy_row[f'Excited State_{i}'] = energy
        rows.append(energy_row)

        # 2. 旋转强度行
        rotatory_row = {
            'ID': mol_id,
            'smiles': smiles,
            'ECD Transition Parameters': 'Rotatory Strength[R(velocity)] (cgs(10**-40 erg-esu-cm/Gauss))'
        }
        for i, R in enumerate(data.y_R.tolist(), 1):
            rotatory_row[f'Excited State_{i}'] = R
        rows.append(rotatory_row)

        # 3. 波长行
        wavelength_row = {
            'ID': mol_id,
            'smiles': smiles,
            'ECD Transition Parameters': 'Wavelengths (nm)'
        }
        if original_mol_id in wavelength_map:
            for i, wl in enumerate(wavelength_map[original_mol_id], 1):
                wavelength_row[f'Excited State_{i}'] = wl
        else:
            missing_wavelengths += 1
            for i in range(1, 21):
                wavelength_row[f'Excited State_{i}'] = None
        rows.append(wavelength_row)

    # 创建并保存
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  成功保存扩展版 CSV: 共 {len(all_data)} 个分子，{len(rows)} 行")
    if missing_wavelengths > 0:
        print(f"  警告: 发现 {missing_wavelengths} 个分子缺失波长数据")


def main():
    """生成对映异构体数据的主流程"""
    args = parse_args()

    print("=" * 80)
    print("CMCDS 对映异构体生成扩展任务启动")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] 正在加载原始数据集...")
    input_dir = Path(args.input_dir)

    train_data = torch.load(input_dir / 'train.pt', weights_only=False)
    val_data = torch.load(input_dir / 'val.pt', weights_only=False)
    test_data = torch.load(input_dir / 'test.pt', weights_only=False)

    print(f"  已从目录加载: {input_dir}")
    print(f"    - train.pt: {len(train_data)} 个分子")
    print(f"    - val.pt:   {len(val_data)} 个分子")
    print(f"    - test.pt:  {len(test_data)} 个分子")
    print(f"  数据集分子总数: {len(train_data) + len(val_data) + len(test_data)}")

    print("\n[2/5] 正在应用空间反射生成对映异构体...")
    print("  训练集 (Train):")
    train_expanded = expand_dataset(train_data, verify=True)
    print("  验证集 (Validation):")
    val_expanded = expand_dataset(val_data, verify=True)
    print("  测试集 (Test):")
    test_expanded = expand_dataset(test_data, verify=True)

    print(f"\n  扩展后的数据集规模 (2x):")
    print(f"    - Train: {len(train_data)} → {len(train_expanded)} 个分子")
    print(f"    - Val:   {len(val_data)} → {len(val_expanded)} 个分子")
    print(f"    - Test:  {len(test_data)} → {len(test_expanded)} 个分子")

    print("\n[3/5] 正在保存扩展后的 .pt 数据集...")
    torch.save(train_expanded, output_dir / 'train.pt')
    torch.save(val_expanded, output_dir / 'val.pt')
    torch.save(test_expanded, output_dir / 'test.pt')
    print(f"  成功保存至: {output_dir}")

    print("\n[4/5] 正在从原始 CSV 加载 Wavelengths 映射...")
    wavelength_map = load_wavelengths_from_csv(args.original_csv)
    print(f"  已成功提取 {len(wavelength_map)} 个分子的波长记录")

    print("\n[5/5] 正在生成并保存扩展后的 CSV 文件...")
    generate_expanded_csv(train_expanded, val_expanded, test_expanded, args.csv_output, wavelength_map)

    print("\n" + "=" * 80)
    print("对映异构体数据生成任务圆满完成！")
    print("=" * 80)

    # 打印一个抽样对比进行确认
    print("\n[示例抽样检查]")
    original = train_data[0]
    enantiomer = train_expanded[1]

    print(f"\n原分子 (ID = {original.mol_id}):")
    print(f"  1号原子坐标: {original.pos[0].numpy()}")
    print(f"  激发态 1 物理量:")
    print(f"    - E (激发能): {original.y_E[0]:.4f} eV")
    print(f"    - μ (电偶极矩): {original.y_mu_vel[0].numpy()}")
    print(f"    - m (磁偶极矩): {original.y_m[0].numpy()}")
    print(f"    - R (旋转强度): {original.y_R[0]:.4e} (10^-40 cgs)")

    print(f"\n对映异构体 (ID = {enantiomer.mol_id}):")
    print(f"  1号原子坐标: {enantiomer.pos[0].numpy()}  <-- Z 轴已反转")
    print(f"  激发态 1 物理量:")
    print(f"    - E (激发能): {enantiomer.y_E[0]:.4f} eV (标量，保持不变)")
    print(f"    - μ (电偶极矩): {enantiomer.y_mu_vel[0].numpy()}  <-- Z 轴已反转")
    print(f"    - m (磁偶极矩): {enantiomer.y_m[0].numpy()}  <-- X, Y 轴已反转")
    print(f"    - R (旋转强度): {enantiomer.y_R[0]:.4e} (10^-40 cgs)  <-- 符号已取反")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()