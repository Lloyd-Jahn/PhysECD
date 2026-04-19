"""
CMCDS数据集的数据准备脚本
==========================================
本脚本将原始CMCDS数据集处理并转换为PyTorch Geometric格式（.pt文件）用于后续图神经网络的训练。

工作流程：
1. 解析CSV文件获取分子ID和标签（激发能E, 旋转强度R）
2. 解析Gaussian输出文件（.log）获取标准坐标系下的3D结构和偶极矩
3. 合并特征与标签数据，并保存为PyG的Data对象
4. 划分为训练集/验证集/测试集
"""

import sys
from pathlib import Path
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import argparse

# 添加父目录到系统路径，以确保能正确导入 physecd 模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physecd.data.parser import GaussianParser
from physecd.data.dataset_cmcds import CMCDSCSVParser


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='准备用于训练的CMCDS数据集')

    parser.add_argument(
        '--csv_path',
        type=str,
        default='data/CMCDS_DATASET.csv',
        help='CMCDS_DATASET.csv文件的路径'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/home/data/jiangyi/Raw_data_gaussian_input_output_V1/Raw_data_gaussian_input_output/Raw_data_gaussian_input_output/Raw_Data_ECD_gaussian_input_output/6.ECD_LOG',
        help='包含Gaussian计算输出.log文件的文件夹路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='处理后的PyG .pt数据文件输出目录'
    )
    parser.add_argument(
        '--n_states',
        type=int,
        default=20,
        help='需要提取的激发态数量（默认：20）'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.9,
        help='训练集比例'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.05,
        help='验证集比例'
    )

    return parser.parse_args()


def process_single_molecule(mol_id, csv_parser, gaussian_parser, n_states=20):
    """
    处理单个分子并创建PyG Data对象。

    参数：
        mol_id: 分子ID
        csv_parser: CMCDSCSVParser实例
        gaussian_parser: GaussianParser实例
        n_states: 激发态数量

    返回：
        PyG Data对象，如果处理失败（例如文件缺失或解析错误）则返回None
    """
    try:
        # 解析CSV数据（获取激发能y_E, 旋转强度y_R, SMILES等）
        csv_data = csv_parser.parse_molecule(mol_id)

        # 解析Gaussian日志文件（获取原子序数z, 坐标pos, 速度电偶极矩y_mu_vel, 磁偶极矩y_m）
        gaussian_data = gaussian_parser.parse_molecule(mol_id, n_states)

        # 创建PyG Data对象，将空间特征和光谱物理量标签绑定在一起
        data = Data(
            z=gaussian_data['z'],                    # [N_atoms] 原子序数
            pos=gaussian_data['pos'],                # [N_atoms, 3] 3D坐标（Standard orientation）
            y_E=csv_data['y_E'],                     # [20] 激发能 (eV)
            y_mu_vel=gaussian_data['y_mu_vel'],      # [20, 3] 速度形式的电偶极矩 (Au)
            y_m=gaussian_data['y_m'],                # [20, 3] 磁偶极矩 (Au)
            y_R=csv_data['y_R'],                     # [20] 速度形式的旋转强度 (10^-40 cgs)
            smiles=csv_data['smiles'],               # SMILES 字符串
            mol_id=mol_id                            # 分子ID
        )

        return data

    except FileNotFoundError:
        # 如果找不到 .log 文件，则跳过该分子
        return None
    except Exception as e:
        print(f"处理分子 {mol_id} 时发生错误: {e}")
        return None


def main():
    """主数据处理流程。"""
    args = parse_args()

    print("=" * 80)
    print("CMCDS 数据集准备流程启动")
    print("=" * 80)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化解析器
    print("\n[1/5] 正在初始化解析器...")
    csv_parser = CMCDSCSVParser(args.csv_path)
    gaussian_parser = GaussianParser(args.log_dir)
    
    print(f"  CSV 解析器已初始化，路径: {args.csv_path}")
    print(f"  Gaussian 日志解析器已初始化")
    print(f"    - LOG 文件夹路径: {args.log_dir}")

    # 从CSV获取所有分子的ID
    print("\n[2/5] 正在从CSV文件中获取分子ID...")
    all_mol_ids = csv_parser.get_all_molecule_ids()
    print(f"  在CSV中共找到 {len(all_mol_ids)} 个分子 (ID 范围: {min(all_mol_ids)}-{max(all_mol_ids)})")

    # 处理所有分子
    print("\n[3/5] 正在处理分子数据...")
    successful_data =[]
    failed_ids =[]

    for mol_id in tqdm(all_mol_ids, desc="  数据提取进度"):
        data = process_single_molecule(mol_id, csv_parser, gaussian_parser, args.n_states)
        if data is not None:
            successful_data.append(data)
        else:
            failed_ids.append(mol_id)

    print(f"\n  成功处理: {len(successful_data)} 个分子")
    print(f"  失败或缺失: {len(failed_ids)} 个分子")

    if len(failed_ids) > 0:
        print(f"  处理失败的分子ID（前20个）: {failed_ids[:20]}")

    if len(successful_data) == 0:
        print("\n错误：没有成功处理任何分子！请检查路径或数据文件格式。")
        return

    # 划分数据集
    print("\n[4/5] 正在划分训练集、验证集和测试集...")
    n_total = len(successful_data)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val

    # 固定随机种子以确保划分结果可复现
    torch.manual_seed(42)
    indices = torch.randperm(n_total).tolist()

    train_data = [successful_data[i] for i in indices[:n_train]]
    val_data = [successful_data[i] for i in indices[n_train:n_train+n_val]]
    test_data = [successful_data[i] for i in indices[n_train+n_val:]]

    print(f"  训练集 (Train): {len(train_data)} 个分子 ({len(train_data)/n_total*100:.1f}%)")
    print(f"  验证集 (Val):   {len(val_data)} 个分子 ({len(val_data)/n_total*100:.1f}%)")
    print(f"  测试集 (Test):  {len(test_data)} 个分子 ({len(test_data)/n_total*100:.1f}%)")

    # 保存处理后的数据集
    print("\n[5/5] 正在保存处理后的数据集 (.pt 文件)...")
    torch.save(train_data, output_dir / 'train.pt')
    torch.save(val_data, output_dir / 'val.pt')
    torch.save(test_data, output_dir / 'test.pt')
    print(f"  已保存至目录: {output_dir}")
    print(f"    - train.pt: 包含 {len(train_data)} 个样本")
    print(f"    - val.pt:   包含 {len(val_data)} 个样本")
    print(f"    - test.pt:  包含 {len(test_data)} 个样本")

    # 打印统计信息
    print("\n" + "=" * 80)
    print("数据集统计信息")
    print("=" * 80)

    sample_data = train_data[0]
    print(f"\n单个样本的数据结构展示 (分子ID: {sample_data.mol_id}):")
    print(f"  z (原子序数):            形状={list(sample_data.z.shape)}, 数据类型={sample_data.z.dtype}")
    print(f"  pos (3D坐标):            形状={list(sample_data.pos.shape)}, 数据类型={sample_data.pos.dtype}")
    print(f"  y_E (激发能):            形状={list(sample_data.y_E.shape)}, 数据类型={sample_data.y_E.dtype}")
    print(f"  y_mu_vel (速度电偶极矩): 形状={list(sample_data.y_mu_vel.shape)}, 数据类型={sample_data.y_mu_vel.dtype}")
    print(f"  y_m (磁偶极矩):          形状={list(sample_data.y_m.shape)}, 数据类型={sample_data.y_m.dtype}")
    print(f"  y_R (旋转强度):          形状={list(sample_data.y_R.shape)}, 数据类型={sample_data.y_R.dtype}")
    print(f"  smiles:                  {sample_data.smiles[:60]}...")

    # 计算全局统计量
    all_num_atoms = [data.z.shape[0] for data in successful_data]
    all_energies = torch.stack([data.y_E for data in successful_data])
    all_R = torch.stack([data.y_R for data in successful_data])

    print(f"\n全局统计信息:")
    print(f"  单分子原子数量:")
    print(f"    - 最小值: {min(all_num_atoms)}")
    print(f"    - 最大值: {max(all_num_atoms)}")
    print(f"    - 平均值: {sum(all_num_atoms)/len(all_num_atoms):.1f}")
    print(f"  激发能 (eV):")
    print(f"    - 范围:[{all_energies.min():.4f}, {all_energies.max():.4f}]")
    print(f"    - 平均值: {all_energies.mean():.4f}")
    print(f"    - 标准差: {all_energies.std():.4f}")
    print(f"  旋转强度 (10^-40 cgs):")
    print(f"    - 范围: [{all_R.min():.4e}, {all_R.max():.4e}]")
    print(f"    - 平均值: {all_R.mean():.4e}")
    print(f"    - 标准差: {all_R.std():.4e}")

    print("\n" + "=" * 80)
    print("数据准备流程已完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()