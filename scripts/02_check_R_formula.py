"""
公式严谨性检查脚本：验证 R = 6414.135151 * (μ_vel · m) / E
=========================================================
本脚本读取已处理好的 .pt 数据，使用代码中的物理公式计算 R (R_calc)，
并与数据集中自带的 R_true (y_R) 进行对比，评估两者的匹配度。
"""

import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='检查物理公式计算的R值与标签R值的匹配度')
    parser.add_argument('--data_path', type=str, 
                        default='data/processed_with_enantiomers/train.pt',
                        help='要检查的 .pt 数据文件路径')
    return parser.parse_args()


def check_formula_consistency(data_path):
    print("=" * 60)
    print(f"正在加载数据: {data_path}")
    
    try:
        dataset = torch.load(data_path, weights_only=False)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print(f"成功加载 {len(dataset)} 个分子样本。")
    print("=" * 60)

    # 收集所有的预测值和真实值
    all_R_calc =[]
    all_R_true = []

    # 用于记录绝对误差
    errors =[]

    for data in dataset:
        y_E = data.y_E             # [20]
        y_mu_vel = data.y_mu_vel   # [20, 3]
        y_m = data.y_m             # [20, 3]
        y_R_true = data.y_R        # [20]

        # R_pred = 6414.135151 * (μ_vel · m) / E
        mu_vel_dot_m = torch.sum(y_mu_vel * y_m, dim=-1)
        
        # 避免除以极小值（和聚合层逻辑保持一致）
        E_safe = torch.clamp(y_E, min=1.0)
        R_calc = 6414.135151 * mu_vel_dot_m / E_safe

        all_R_calc.append(R_calc)
        all_R_true.append(y_R_true)
        
        # 计算绝对误差
        errors.append(torch.abs(R_calc - y_R_true))

    # 展平所有的张量
    R_calc_tensor = torch.cat(all_R_calc)
    R_true_tensor = torch.cat(all_R_true)
    errors_tensor = torch.cat(errors)

    # 计算统计指标
    mae = torch.mean(errors_tensor).item()
    mse = torch.mean(errors_tensor ** 2).item()
    max_error = torch.max(errors_tensor).item()

    # 计算皮尔逊相关系数
    vx = R_calc_tensor - torch.mean(R_calc_tensor)
    vy = R_true_tensor - torch.mean(R_true_tensor)
    correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    print("\n【全局统计结果】")
    print(f"样本总数 (Excited States): {len(R_calc_tensor)}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"均方误差     (MSE): {mse:.4f}")
    print(f"最大绝对误差 (Max Error): {max_error:.4f}")
    print(f"皮尔逊相关系数 (PCC): {correlation.item():.6f} (越接近1说明线性关系越完美)")

    print("\n【抽样详细对比 (前10个激发态)】")
    print(f"{'State':<10} | {'R_true':<15} | {'R_calc':<18} | {'绝对误差':<10}")
    print("-" * 60)
    for i in range(10):
        t_val = R_true_tensor[i].item()
        c_val = R_calc_tensor[i].item()
        err = errors_tensor[i].item()
        print(f"State {i+1:<4} | {t_val:>15.4f} | {c_val:>18.4f} | {err:>10.4f}")
    
    print("=" * 60)
    
    # 诊断结论
    print("\n【诊断结论】")
    if correlation.item() > 0.99:
        print("-> 结论：皮尔逊相关系数极高，说明公式正确。")
        print("-> 如果存在小的绝对误差，大概率是因为 Gaussian .log 文件输出的 μ_vel 和 m")
        print("   仅仅保留了 4 位小数（例如 0.0268），存在数值截断误差。")
    else:
        print("-> 结论：相关系数不理想，公式本身或单位换算常数错误。")


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    check_formula_consistency(args.data_path)