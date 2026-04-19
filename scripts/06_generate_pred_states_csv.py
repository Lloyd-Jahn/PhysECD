"""
导出分子的预测激发能和旋转强度。

输出包含20个激发态的CSV文件，内容包括：
- 激发态序号
- 激发能（eV）
- 波长（nm）
- 旋转强度 R（10^-40 cgs）
- 真实值用于对比
"""

"""
运行指令：
  cd到PhysECD目录下，执行以下命令：
  python scripts/06_generate_pred_states_csv.py --mol_id x
  x是分子ID，可以自己指定
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import argparse

# 添加父目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physecd.models import PhysECDModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export predicted excitation energies and rotatory strengths'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/processed_with_enantiomers/test.pt',
                        help='Path to test dataset')
    parser.add_argument('--mol_id', type=int, required=True,
                        help='Molecule ID to predict (e.g. 6283 or -6283)')
    parser.add_argument('--output_dir', type=str, default='ecd_pred_results',
                        help='Output directory for CSV files')
    return parser.parse_args()


def load_model_and_data(checkpoint_path, test_data_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['config']

    model = PhysECDModel(
        num_features=config['num_features'],
        max_l=config['max_l'],
        num_blocks=config['num_blocks'],
        num_radial=config['num_radial'],
        cutoff=config['cutoff'],
        n_states=config['n_states'],
        max_atomic_number=config['max_atomic_number']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Model loaded (epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.6f})")

    print(f"Loading test data from: {test_data_path}")
    test_data = torch.load(test_data_path, weights_only=False)
    print(f"  Test set size: {len(test_data)} molecules")

    return model, test_data, config


def run_inference(model, data, device):
    from torch_geometric.data import Batch
    if getattr(data, 'batch', None) is None:
        data = Batch.from_data_list([data])

    model = model.to(device)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    E_pred = output['E_pred'][0].cpu().numpy()   # [20]
    R_pred = output['R_pred'][0].cpu().numpy()    # [20]，单位为 10^-40 cgs

    return E_pred, R_pred


def main():
    args = parse_args()

    print("=" * 80)
    print("Export Predicted Excited States")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型和数据
    model, test_data, config = load_model_and_data(args.checkpoint, args.test_data)

    # 按分子ID查找分子
    data = None
    for sample in test_data:
        mid = sample.mol_id.item() if hasattr(sample.mol_id, 'item') else int(sample.mol_id)
        if mid == args.mol_id:
            data = sample
            break
    if data is None:
        raise ValueError(f"Molecule ID {args.mol_id} not found in test set")

    mol_id = args.mol_id
    print(f"\nMolecule ID: {mol_id}")
    print(f"Number of atoms: {data.z.shape[0]}")
    print(f"SMILES: {data.smiles}")

    # 运行推理
    print("\nRunning inference...")
    E_pred, R_pred = run_inference(model, data, device)

    # 获取真实值
    E_true = data.y_E.numpy().flatten()    # [20]
    R_true = data.y_R.numpy().flatten()    # [20]，单位为 10^-40 cgs

    # 构建CSV
    wavelength_pred = 1240.0 / E_pred   # nm
    wavelength_true = 1240.0 / E_true   # nm

    df = pd.DataFrame({
        'State': np.arange(1, 21),
        'E_pred (eV)': np.round(E_pred, 4),
        'E_true (eV)': np.round(E_true, 4),
        'E_error (eV)': np.round(E_pred - E_true, 4),
        'λ_pred (nm)': np.round(wavelength_pred, 2),
        'λ_true (nm)': np.round(wavelength_true, 2),
        'R_pred (10^-40 cgs)': np.round(R_pred, 4),
        'R_true (10^-40 cgs)': np.round(R_true, 4),
        'R_error (10^-40 cgs)': np.round(R_pred - R_true, 4),
    })

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{mol_id}_excited_states.csv"
    df.to_csv(output_path, index=False)

    # 打印表格
    print(f"\n{'State':>5} {'E_pred':>10} {'E_true':>10} {'ΔE':>10} {'R_pred':>14} {'R_true':>14} {'ΔR':>14}")
    print("-" * 80)
    for i in range(20):
        print(f"{i+1:>5} {E_pred[i]:>10.4f} {E_true[i]:>10.4f} {E_pred[i]-E_true[i]:>10.4f} "
              f"{R_pred[i]:>14.4f} {R_true[i]:>14.4f} {R_pred[i]-R_true[i]:>14.4f}")

    # 汇总统计
    mae_E = np.abs(E_pred - E_true).mean()
    mae_R = np.abs(R_pred - R_true).mean()
    print(f"\nMAE(E) = {mae_E:.4f} eV")
    print(f"MAE(R) = {mae_R:.4f} (10^-40 cgs)")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
