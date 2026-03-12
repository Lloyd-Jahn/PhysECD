# PhysECD 项目介绍文档

## 文档说明
本文档面向 Claude Code，旨在帮助理解 PhysECD 项目的完整工作流程、模型架构、物理原理和代码实现。通过阅读本文档，Claude Code 能够更好地理解项目背景，从而高效完成具体的开发任务。

---

## 1. 项目概述

### 1.1 研究目标
PhysECD 是一个**物理驱动的深度学习框架**，用于从手性有机分子的三维结构预测其电子圆二色谱（Electronic Circular Dichroism, ECD）。

与现有方法（如 ECDFormer）将光谱预测视为"黑盒序列生成"不同，PhysECD 采用**第一性原理驱动**的方法：
- 使用 **SE(3) 等变图神经网络**提取分子的几何特征
- 预测**原子级别**的量子跃迁贡献（电荷、偶极矩、电流）
- 通过**严格的量子化学公式**聚合为分子级别的物理量
- 最终计算出决定 ECD 光谱峰值的**旋转强度（Rotatory Strength, R）**

### 1.2 核心创新点
1. **SE(3) 等变性**：模型对分子的旋转和平移保持不变性，符合物理规律
2. **原子级预测 + 物理聚合**：不直接预测宏观量，而是预测微观贡献后用物理公式聚合
3. **无参数物理层**：聚合层完全基于量子化学公式，无需学习参数
4. **多任务学习**：同时监督激发能、电偶极矩、磁偶极矩和旋转强度

---

## 2. 数据集与数据处理

### 2.1 CMCDS 数据集
- **来源**：TD-DFT（Time-Dependent Density Functional Theory）计算
- **分子数量**：10,887 个手性有机分子
- **数据文件**：
  - `CMCDS_DATASET.csv`：包含每种分子前 20 个激发态激发能（E）、旋转强度（R）、波长、SMILES
  - `.gjf` 文件：包含 DFT 优化后的 3D 坐标
  - `.log` 文件：包含速度电偶极矩（μ_vel）和磁偶极矩（m）

### 2.2 数据准备流程（`01_prepare_data.py`）

**输入**：
- CSV 文件：`CMCDS_DATASET.csv`
- Gaussian 文件：`.gjf`（坐标）和 `.log`（偶极矩）

**处理步骤**：
1. **解析 CSV**：提取每个分子的 20 个激发态的激发能（E）和旋转强度（R）
2. **解析 .gjf 文件**：提取原子坐标（pos）和原子序数（z）
3. **解析 .log 文件**：提取速度电偶极矩（μ_vel）和磁偶极矩（m）
4. **创建 PyG Data 对象**：将所有数据整合为 PyTorch Geometric 格式
5. **数据集划分**：按 8:1:1 比例划分为训练集、验证集、测试集

**输出**：
- `data/processed/train.pt`：8,709 个分子
- `data/processed/val.pt`：1,088 个分子
- `data/processed/test.pt`：1,090 个分子

**PyG Data 对象结构**：
```python
Data(
    z=[N_atoms],              # 原子序数 (int64)
    pos=[N_atoms, 3],         # 3D 坐标 (float32, 单位: Å)
    y_E=[20],                 # 激发能 (float32, 单位: eV)
    y_mu_vel=[20, 3],         # 速度电偶极矩 (float32, 单位: a.u.)
    y_m=[20, 3],              # 磁偶极矩 (float32, 单位: a.u.)
    y_R=[20],                 # 旋转强度 (float32, 单位: 10^-40 cgs)
    smiles=str,               # SMILES 字符串
    mol_id=int                # 分子 ID
)
```

### 2.3 对映体生成（`02_generate_enantiomers.py`）

**物理原理**：
手性分子存在对映异构体（Enantiomers），即镜像对称的分子。对映体具有以下性质：
- **几何关系**：互为镜像（通过平面反射变换）
- **ECD 光谱关系**：旋转强度符号相反（R → -R），光谱形状镜像对称

**数据增强策略**：
通过几何变换生成对映体，无需额外的 DFT 计算即可将数据集扩充 2 倍。

**变换公式**（以 XY 平面为镜面）：
1. **坐标变换**：$(x, y, z) \rightarrow (x, y, -z)$
2. **电偶极矩**（极矢量）：$(\mu_x, \mu_y, \mu_z) \rightarrow (\mu_x, \mu_y, -\mu_z)$
3. **磁偶极矩**（赝矢量）：$(m_x, m_y, m_z) \rightarrow (-m_x, -m_y, m_z)$
4. **旋转强度**：$R \rightarrow -R$
5. **激发能**：保持不变

**验证**：
通过检查 $\vec{\mu}' \cdot \vec{m}' = -\vec{\mu} \cdot \vec{m}$ 来验证变换的正确性。

**输出**：
- `/home/data/jiangyi/PhysECD/data/CMCDS_DATASET_with_enantiomers.csv`：包含 21774 种分子（10887 * 2）前 20 个激发态的激发能（E）、旋转强度（R）、波长、SMILES
- `data/processed_with_enantiomers/train.pt`：17,418 个分子（原始 + 对映体）
- `data/processed_with_enantiomers/val.pt`：2,176 个分子
- `data/processed_with_enantiomers/test.pt`：2,180 个分子

---

## 3. 模型架构

PhysECD 模型由三个核心组件构成：

### 3.1 SE(3) 等变主干网络（`physecd/models/se3_backbone.py`）

**功能**：从分子图中提取 SE(3) 等变特征

**输入**：
- `z`: [N_atoms] - 原子序数
- `pos`: [N_atoms, 3] - 原子坐标
- `batch`: [N_atoms] - batch 索引

**架构**：
1. **原子嵌入层**：将原子序数映射为初始特征向量
2. **径向基函数**：编码原子间距离信息
3. **球谐函数**：编码原子间方向信息（保持 SO(3) 等变性）
4. **交互块**（Interaction Blocks）：
   - 使用消息传递机制更新节点特征
   - 包含标量特征（S）和张量特征（T）
   - 通过多头注意力机制聚合邻居信息

**输出**：
- `S`: [N_atoms, num_features] - 标量特征
- `T`: [N_atoms, irreps_T.dim] - 等变张量特征

### 3.2 多任务预测头（`physecd/models/heads.py`）

**功能**：从主干网络特征预测原子级别的量子性质

**对于每个原子`A`，都有 5 个预测头**：

1. **激发能头（E_pred）**：
   - 输入：标量特征 S
   - 输出：[Batch_size, 20] - 20 个激发态的能量
   - 方法：全局平均池化 + MLP

2. **原子跃迁电荷头（q_A）**：
   - 输入：标量特征 S
   - 输出：[N_atoms, 20] - 每个原子在 20 个激发态的跃迁电荷
   - 方法：MLP（原子级别，无池化）

3. **原子电偶极矩头（μ_A）**：
   - 输入：张量特征 T
   - 输出：[N_atoms, 20, 3] - 每个原子的局域电偶极矩
   - 方法：e3nn.o3.Linear 投影到 1o 不可约表示（极矢量）

4. **原子磁偶极矩头（m_A）**：
   - 输入：张量特征 T
   - 输出：[N_atoms, 20, 3] - 每个原子的局域磁偶极矩
   - 方法：e3nn.o3.Linear 投影到 1e 不可约表示（赝矢量）

5. **原子跃迁电流头（v_A）**：
   - 输入：张量特征 T
   - 输出：[N_atoms, 20, 3] - 每个原子的跃迁电流强度
   - 方法：e3nn.o3.Linear 投影到 1o 不可约表示（极矢量）

### 3.3 物理聚合层（`physecd/physics/aggregation.py`）

**功能**：将原子级预测聚合为分子级物理量（**无学习参数，纯物理公式**）

**输入**：
- `pos`: [N_atoms, 3] - 原子坐标
- `q_A`: [N_atoms, 20] - 原子跃迁电荷
- `μ_A`: [N_atoms, 20, 3] - 原子电偶极矩
- `m_A`: [N_atoms, 20, 3] - 原子磁偶极矩
- `v_A`: [N_atoms, 20, 3] - 原子跃迁电流

**处理步骤**：

**步骤 1：坐标去中心化**
```
pos_delta = pos - mean(pos)
```
去中心化不仅保证了 SE(3) 中的平移不变性，同时在数学上自动满足了跃迁电荷的规范不变性。

**步骤 2：电偶极矩聚合**
```
μ_total = Σ_A[μ_A + q_corrected,A * pos_delta]
```
- 第一项：原子的局域电偶极矩
- 第二项：电荷分布产生的偶极矩贡献

**步骤 3：磁偶极矩聚合**
```
m_total = Σ_A[m_A + 0.5 * (pos_delta × v_corrected,A)]
```
- 第一项：原子的局域磁偶极矩
- 第二项：轨道角动量产生的磁偶极矩（叉乘）

**步骤 4：旋转强度计算**
```
R_pred = 235.715 · (μ_total · m_total) / E_pred
```
- 点积计算旋转强度（单位：a.u.）

**输出**：
- `μ_total`: [Batch_size, 20, 3] - 分子总电偶极矩
- `m_total`: [Batch_size, 20, 3] - 分子总磁偶极矩
- `R_pred`: [Batch_size, 20] - 旋转强度
---

## 4. 训练流程

### 4.1 损失函数（`physecd/physics/loss.py`）

**多任务 MSE 损失**：
```
L_total = λ_E · L_E + λ_μ · L_μ + λ_m · L_m + λ_R · L_R
```

**各项损失**：
1. **激发能损失**：
   ```
   L_E = MSE(E_pred, E_true)
   ```

2. **电偶极矩损失**：
   ```
   L_μ = MSE(μ_total, μ_vel,true)
   ```

3. **磁偶极矩损失**：
   ```
   L_m = MSE(m_total, m_true)
   ```

4. **旋转强度损失**（关键：单位对齐）：
   ```
   R_pred_scaled = R_pred × 471.44  # a.u. → 10^-40 cgs
   L_R = MSE(R_pred_scaled, R_true)
   ```

**单位转换说明**：
- 模型预测的 R_pred 单位是原子单位（a.u.）
- 数据集中的 R_true 单位是 10^-40 cgs
- 转换系数：1 a.u. = 471.44 × 10^-40 cgs

**权重设置（后续还需要调整）**：
- `λ_E = 1.0`：激发能容易学习
- `λ_μ = 10.0`：电偶极矩需要较高权重
- `λ_m = 50.0`：磁偶极矩最难学习，给予最高权重
- `λ_R = 0.1`：旋转强度已经放大 471.44 倍，权重降低

### 4.2 训练脚本（`scripts/04_train.py`）

**训练配置**：
```python
config = {
    'batch_size': 32,
    'num_epochs': 25,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'num_features': 128,
    'max_l': 2,
    'num_blocks': 3,
    'num_radial': 32,
    'cutoff': 5.0,
    'n_states': 20,
}
```
---

## 5. 光谱生成与可视化

### 5.1 光谱生成（`scripts/06_generate_spectrum.py`）

**目标**：从模型预测的离散激发态生成连续的 ECD 光谱曲线

**输入**：
- 训练好的模型（`checkpoints/best_model.pt`）
- 测试集分子（`data/processed_with_enantiomers/test.pt`）

**步骤 1：模型推理**
```python
E_pred, R_pred_au = model(data)
# E_pred: [20] 激发能 (eV)
# R_pred_au: [20] 旋转强度 (a.u.)
```

**步骤 2：单位转换**
```python
R_cgs = R_pred_au × 471.44  # a.u. → 10^-40 cgs
```

**步骤 3：波长-能量转换**
```python
λ = [80, 81, ..., 450] nm  # 波长网格
E_grid = 1240 / λ  # eV
```

**步骤 4：高斯展宽**

将离散的激发态展宽为连续光谱：
```
Δε(E) = (1 / (2.296×10¹ × σ × √π)) × Σ_i E_i × R_cgs,i × exp[-(E - E_i)² / σ²]
```

参数说明：
- `σ = 0.4 eV`：高斯展宽标准差（该参数后期可以调整）
- `2.296×10¹`：归一化常数（考虑 R_cgs 的单位）

**步骤 5：转换为摩尔椭圆度（[θ]）**
```
[θ] = Δε * 3298.2
```

**输出**：
- CSV 文件：`ecd_pred_results/{mol_id}_predicted.csv`
  - 列 1：Wavelength (nm)
  - 列 2：[θ]

### 5.2 光谱绘图（`scripts/07_plot_spectrum.py`）

**功能**：从 CSV 文件生成高质量的 ECD 光谱图

**输入**：
- CSV 文件（包含波长和 ECD 值）

**输出**：
- PNG 图像（300 DPI，适合发表）
- 包含基线（y=0）、网格、图例
---

## 6. 完整工作流程总结

### 6.1 数据准备阶段
```bash
# 步骤 1：处理原始数据
python scripts/01_prepare_data.py

# 步骤 2：生成对映体数据（数据增强）
python scripts/02_generate_enantiomers.py

# 步骤 3：验证数据集
python scripts/03_validate_data.py
```

### 6.2 模型训练阶段
```bash
# 训练模型
python scripts/04_train.py
```

### 6.3 光谱预测阶段
```bash
# 步骤 1：生成光谱数据
python scripts/06_generate_spectrum.py --mol_idx X

# 步骤 2：绘制光谱图
python scripts/07_plot_spectrum.py --csv_path ecd_pred_results/XXXX_predicted.csv
```

---

## 7. 项目文件结构

```
PhysECD/
├── data/                           # 数据目录
│   ├── processed/                  # 原始处理后的数据
│   └── processed_with_enantiomers/ # 包含对映体的数据
│
├── physecd/                        # 核心代码包
│   ├── data/                       # 数据处理模块
│   │   ├── parser.py               # Gaussian 文件解析器
│   │   └── dataset_cmcds.py        # CSV 解析器
│   ├── models/                     # 模型模块
│   │   ├── se3_backbone.py         # SE(3) 主干网络
│   │   ├── heads.py                # 多任务预测头
│   │   ├── physecd_model.py        # 完整模型
│   │   └── modules/                # 基础模块（嵌入、MLP、注意力等）
│   └── physics/                    # 物理层模块
│       ├── aggregation.py          # 物理聚合层
│       └── loss.py                 # 损失函数
│
├── scripts/                        # 执行脚本
│   ├── 01_prepare_data.py          # 数据准备
│   ├── 02_generate_enantiomers.py  # 对映体生成
│   ├── 03_validate_data.py         # 数据验证
│   ├── 04_train.py                 # 模型训练
│   ├── 06_generate_spectrum.py     # 光谱生成
│   └── 07_plot_spectrum.py         # 光谱绘图
│
├── checkpoints/                    # 模型检查点
├── ecd_pred_results/               # 预测结果
└── prompts_and_work_progress/      # 项目介绍文档与Claude Code提示词
```