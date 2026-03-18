"""
读取 train.pt 文件中第 x 个分子的所有数据
用法：python 01_read_train_pt.py --index x
"""

import torch
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='读取 train.pt 中的指定分子数据')
    parser.add_argument('--file_path', type=str, default='/home/data/jiangyi/PhysECD-3.17修改-公式严谨性检查+单独训练R/data/processed_with_enantiomers/train.pt',
                        help='train.pt 文件的路径')
    parser.add_argument('--index', type=int, required=True,
                        help='要读取的分子索引（从0开始）')
    return parser.parse_args()


def print_data_structure(data, mol_index):
    """打印 Data 对象的所有属性和内容"""
    print("=" * 80)
    print(f"分子索引: {mol_index}")
    print("=" * 80)

    # 获取所有属性（排除私有方法和属性）
    attributes = [key for key in data.keys() if not key.startswith('_')]

    for key in attributes:
        value = getattr(data, key)
        print(f"\n[{key}]")
        if isinstance(value, torch.Tensor):
            print(f"  类型: torch.Tensor")
            print(f"  形状: {value.shape}")
            print(f"  数据类型: {value.dtype}")
            print(f"  设备: {value.device}")
            print(f"  数值:")
            print(value)
        elif isinstance(value, str):
            print(f"  类型: str")
            print(f"  长度: {len(value)}")
            print(f"  内容: {value}")
        elif isinstance(value, (int, float)):
            print(f"  类型: {type(value).__name__}")
            print(f"  值: {value}")
        elif value is None:
            print("  值为 None")
        else:
            print(f"  类型: {type(value).__name__}")
            print(f"  值: {value}")


def main():
    args = parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    print(f"正在加载文件: {file_path}")
    data_list = torch.load(file_path)

    if not isinstance(data_list, list):
        raise TypeError("文件内容不是列表格式，可能不是有效的 train.pt 文件")

    total = len(data_list)
    print(f"文件中包含 {total} 个分子")

    idx = args.index
    if idx < 0 or idx >= total:
        raise IndexError(f"索引 {idx} 超出范围，有效范围为 0 ~ {total - 1}")

    data = data_list[idx]
    print_data_structure(data, idx)


if __name__ == "__main__":
    main()