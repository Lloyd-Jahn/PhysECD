# PhysECD 项目介绍文档

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
4. **多任务学习**：同时监督激发能（E）、速度电偶极矩（μ_vel）和旋转强度（R）；μ_vel 与磁偶极矩（m）通过**联合相位一致性损失**共同约束（`lambda_mu_vel=1.0, lambda_m=1.0`），确保两者选择同一量子相位

---

## 2. 数据集与数据处理

### 2.1 CMCDS 数据集
- **来源**：TD-DFT（Time-Dependent Density Functional Theory）计算
- **分子数量**：10,887 个手性有机分子
- **数据文件**：
  - `CMCDS_DATASET.csv`：包含每种分子前 20 个激发态激发能（E）、旋转强度（R）、波长、SMILES
  - `.log` 文件：包含 DFT 优化后的 3D 坐标、速度电偶极矩（μ_vel）和磁偶极矩（m）

### 2.2 数据准备流程（`scripts/01_prepare_data.py`）

**输入**：
- CSV 文件：`CMCDS_DATASET.csv`
- Gaussian 文件：`.log`

**处理步骤**：
1. **解析 CSV**：提取每个分子的 20 个激发态的激发能（E）和旋转强度（R）
2. **解析 .log 文件**：提取原子坐标（pos）和原子序数（z）、速度电偶极矩（μ_vel）和磁偶极矩（m）
3. **创建 PyG Data 对象**：将所有数据整合为 PyTorch Geometric 格式
4. **数据集划分**：固定随机种子（seed=42），按 8:1:1 比例划分为训练集、验证集、测试集

**输出**：
- `data/processed/train.pt`：8,709 个分子
- `data/processed/val.pt`：1,088 个分子
- `data/processed/test.pt`：1,090 个分子

**PyG Data 对象结构**：
```python
Data(
    z=[N_atoms],              # 原子序数 (int64)
    pos=[N_atoms, 3],         # 3D 坐标 (float32, 单位: Å，Standard orientation)
    y_E=[20],                 # 激发能 (float32, 单位: eV)
    y_mu_vel=[20, 3],         # 速度电偶极矩 (float32, 单位: a.u.)
    y_m=[20, 3],              # 磁偶极矩 (float32, 单位: a.u.)
    y_R=[20],                 # 旋转强度 (float32, 单位: 10^-40 cgs)
    smiles=str,               # SMILES 字符串
    mol_id=int                # 分子 ID（正整数）
)
```

### 2.3 公式严谨性检查（`scripts/02_check_formula.py`）

**功能**：验证物理公式 `R = 6414.135151 × (μ_vel · m) / E` 与数据集标签 `y_R` 之间的一致性，是一次性验证脚本。

**运行方式**：
```bash
cd /home/data/jiangyi/PhysECD-3.23-初步收敛但过拟合
python scripts/02_check_formula.py --data_path data/processed_with_enantiomers/train.pt
```

**验证逻辑**：
- 读取已处理好的 `.pt` 数据，直接使用数据集中的 `y_E`、`y_mu_vel`、`y_m` 套入 `aggregation.py` 的公式计算 `R_calc`
- 将 `R_calc` 与标签 `y_R` 对比，输出 MAE、最大绝对误差和皮尔逊相关系数

**诊断标准**：
- 相关系数 > 0.99：公式形式正确，残余误差来自 Gaussian `.log` 文件的 4 位小数截断误差
- 相关系数 < 0.99：公式常数 `6414.135151` 或点积方向存在问题

### 2.4 对映体生成（`scripts/02_generate_enantiomers.py`）

**物理原理**：
手性分子存在对映异构体（Enantiomers），即镜像对称的分子。对映体具有以下性质：
- **几何关系**：互为镜像（通过平面反射变换）
- **ECD 光谱关系**：旋转强度符号相反（R → -R），光谱形状镜像对称

**数据增强策略**：
通过几何变换生成对映体，无需额外的 DFT 计算即可将数据集扩充 2 倍。

**变换公式**（以 XY 平面为镜面，即 z → -z）：
1. **坐标变换**：$(x, y, z) \rightarrow (x, y, -z)$
2. **速度电偶极矩**（极矢量）：$(\mu_{vel,x}, \mu_{vel,y}, \mu_{vel,z}) \rightarrow (\mu_{vel,x}, \mu_{vel,y}, -\mu_{vel,z})$
3. **磁偶极矩**（伪矢量）：$(m_x, m_y, m_z) \rightarrow (-m_x, -m_y, m_z)$
4. **旋转强度**：$R \rightarrow -R$
5. **激发能**：保持不变（标量）
6. **SMILES 手性标识**：`@` 与 `@@` 互换（反转立体化学标注）

**对映体标识**：对映体的 `mol_id` 取原始分子 ID 的相反数（如原始为 `6283`，对映体为 `-6283`）。

**验证**：
通过检查 $\vec{\mu}_{vel}' \cdot \vec{m}' = -\vec{\mu}_{vel} \cdot \vec{m}$ 来验证变换的正确性（脚本自动计算并输出最大误差）。

**输出**：
- `data/CMCDS_DATASET_with_enantiomers.csv`：包含 21,774 种分子（10,887 × 2）前 20 个激发态的激发能（E）、旋转强度（R）、波长、SMILES
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
1. **原子嵌入层**：将原子序数映射为初始标量特征向量（Swish 激活）
2. **径向基函数**：使用**可训练 Bessel 基函数**（`trainable_bessel`）编码原子间距离；带截断函数，截断半径 `cutoff=50.0 Å`
3. **球谐函数**：将归一化的原子间方向向量展开为球谐函数（`lmax=max_l=3`），编码方向信息并保持 SO(3) 等变性
4. **交互块**（Interaction Blocks，共 `num_blocks=3` 个）：
   - 消息传递机制更新节点的标量特征（S）和等变张量特征（T）
   - **8 头注意力**聚合邻居信息
   - 近距离边过滤：原子间距 < 0.3 Å 的边被丢弃（避免数值不稳定）
   - 每次 batch 构图时邻居数上限 `max_num_neighbors=128`

**等变张量特征 `irreps_T` 的结构**（固定，与 `max_l` 无关）：
```
irreps_T = [
    (num_features, 1o),   # 极矢量（polar vectors），用于预测速度电偶极矩
    (num_features, 1e),   # 赝矢量（pseudo-vectors），用于预测磁偶极矩
    (num_features, 2e),   # 二阶赝张量（pseudo-tensors）
]
```

**输出**：
- `S`: [N_atoms, num_features] - 标量特征（旋转不变）
- `T`: [N_atoms, irreps_T.dim] - 等变张量特征（随旋转协变变换）

### 3.2 多任务预测头（`physecd/models/heads.py`）

**功能**：从主干网络特征分别预测分子级激发能和原子级量子性质

共有 **5 个预测头**，其中激发能头为**分子级**（全局池化），其余 4 个为**原子级**（无池化）：

1. **激发能头（E_pred）**——分子级：
   - 输入：标量特征 S
   - 流程：MLP → `global_mean_pool`（原子→分子）→ Linear
   - 输出：[Batch_size, 20] - 20 个激发态的能量
   - **物理约束**：`E_pred = softplus(E_raw) + 3.0`，确保预测值 ≥ 3.0 eV（排除非物理的低能激发态）

2. **原子跃迁电荷头（q_A）**——原子级：
   - 输入：标量特征 S
   - 输出：[N_atoms, 20] - 每个原子在 20 个激发态的跃迁电荷
   - 方法：MLP（无池化）

3. **原子速度电偶极矩头（`mu_A_vel`）**——原子级：
   - 输入：张量特征 T
   - 输出：[N_atoms, 20, 3] - 每个原子的局域速度电偶极矩
   - 方法：e3nn.o3.Linear 投影到 **1o** 不可约表示（极矢量）

4. **原子磁偶极矩头（m_A）**——原子级：
   - 输入：张量特征 T
   - 输出：[N_atoms, 20, 3] - 每个原子的局域磁偶极矩
   - 方法：e3nn.o3.Linear 投影到 **1e** 不可约表示（赝矢量）

5. **原子跃迁电流头（v_A）**——原子级：
   - 输入：张量特征 T
   - 输出：[N_atoms, 20, 3] - 每个原子的跃迁电流强度
   - 方法：e3nn.o3.Linear 投影到 **1o** 不可约表示（极矢量）

### 3.3 物理聚合层（`physecd/physics/aggregation.py`）

**功能**：将原子级预测聚合为分子级物理量（**无学习参数，纯物理公式**）

**输入**：
- `pos`: [N_atoms, 3] - 原子坐标
- `q_A`: [N_atoms, 20] - 原子跃迁电荷
- `mu_A_vel`: [N_atoms, 20, 3] - 原子速度电偶极矩
- `m_A`: [N_atoms, 20, 3] - 原子磁偶极矩
- `v_A`: [N_atoms, 20, 3] - 原子跃迁电流

**处理步骤**：

**步骤 1：坐标去中心化**
```
pos_delta = pos - scatter_mean(pos)
```
保证平移不变性，并自动满足跃迁电荷的规范不变性。

**步骤 2：速度电偶极矩聚合**
```
mu_total_vel = Σ_A[mu_A_vel + q_A * pos_delta]
```
- 第一项：原子的局域速度电偶极矩
- 第二项：电荷分布产生的偶极矩贡献

**步骤 3：磁偶极矩聚合**
```
m_total = Σ_A[m_A + 0.5 * (pos_delta × v_A)]
```
- 第一项：原子的局域磁偶极矩
- 第二项：轨道角动量产生的磁偶极矩（叉乘）

**步骤 4：旋转强度计算**
```
R_pred = 6414.135151 · (mu_total_vel · m_total) / E_pred
```
- 点积计算旋转强度，输出直接是 **10^-40 cgs 单位**（与数据集标签单位一致，无需额外换算）
- 数值稳定性：`E_pred` 被 clamp 到 `min=1.0 eV` 防止除零

**输出**：
- `mu_total_vel`: [Batch_size, 20, 3] - 分子总速度电偶极矩（a.u.）
- `m_total`: [Batch_size, 20, 3] - 分子总磁偶极矩（a.u.）
- `R_pred`: [Batch_size, 20] - 旋转强度（10^-40 cgs，已换算）

---

## 4. 训练流程

### 4.1 损失函数（`physecd/physics/loss.py`）

**多任务损失（EMA 动态归一化 + 联合 μ-m 相位一致性）**：
```
L_total = λ_E · norm_E  +  λ_μ_vel · norm_μm_joint  +  λ_R · norm_R
```

其中各归一化分量定义如下：
```
norm_E        = L_E / EMA_E
norm_μm_joint = min(norm_μ_phase1 + norm_m_phase1,   # 两者同相位
                    norm_μ_phase2 + norm_m_phase2)    # 两者反相位
norm_R        = L_R / EMA_R
```

各分量通过指数移动平均（EMA，decay=0.99）追踪其量级并动态归一化。

**联合 μ-m 相位一致性损失（核心设计）**：

μ_vel 和 m 来自同一量子波函数，共享相同的全局相位符号不定性——它们必须同时翻转（同号或同时取反），而非各自独立选择相位。旧版本对两者分别独立取最小值，可能使 μ 选正相、m 选负相，违反物理约束。新版本将两者归一化后**先相加再取最小**：

```
norm_μ_phase1 = mean(‖μ_pred - μ_true‖²) / EMA_μ       # μ 正相
norm_μ_phase2 = mean(‖μ_pred + μ_true‖²) / EMA_μ       # μ 反相
norm_m_phase1 = mean(‖m_pred - m_true‖²) / EMA_m       # m 正相
norm_m_phase2 = mean(‖m_pred + m_true‖²) / EMA_m       # m 反相

norm_μm_joint = min(norm_μ_phase1 + norm_m_phase1,
                    norm_μ_phase2 + norm_m_phase2)
```

**各项损失**：
1. **激发能损失**：
   ```
   L_E = MSE(E_pred, E_true)
   ```

2. **旋转强度损失（L1 Loss）**：
   ```
   L_R = L1(R_pred, R_true)
   ```
   R_pred 与 R_true 均使用 10^-40 cgs 单位。使用 L1 而非 MSE 以提高对极端值的鲁棒性。

**辅助指标**（仅监控，不参与反向传播）：
- `R_sign_acc`：旋转强度符号预测准确率
- `loss_mu_vel`、`loss_m`：各自独立取最小值后的原始损失（用于帕累托前沿监控）

**权重设置**：
```python
lambda_E       = 1.0   # 激发能
lambda_mu_vel  = 1.0   # 联合 μ-m 相位损失（同时约束 μ_vel 和 m）
lambda_m       = 1.0   # 保留参数，联合约束中已隐含，不再单独加权
lambda_R       = 1.0   # 旋转强度
lambda_R_sign  = 0.0   # 符号分类辅助损失（当前关闭）
```

**`loss_dict` 返回键**：
```
loss, loss_E, loss_mu_vel, loss_m, loss_R,
loss_R_l1, loss_R_sign, R_sign_acc,
norm_E, norm_mu_m, norm_R
```

### 4.2 训练脚本（`scripts/03_train.py`）

**训练配置**：
```python
config = {
    'data_dir': 'data/processed_with_enantiomers',  # 使用含对映体的增强数据集
    'batch_size': 64,
    'num_epochs': 1000,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'num_workers': 16,
    # 模型超参数
    'num_features': 128,
    'max_l': 3,           # 球谐函数最高阶
    'num_blocks': 3,
    'num_radial': 32,
    'cutoff': 50.0,       # 较大截断半径，覆盖整个分子
    'n_states': 20,
    'max_atomic_number': 60,
    # 损失权重
    'lambda_E': 1.0,
    'lambda_mu_vel': 1.0,
    'lambda_m': 0.0,
    'lambda_R': 1.0,
    'lambda_R_sign': 0.0,
}
```

**优化器与学习率调度**：
```python
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=num_epochs//5)
```
- 余弦退火，每 200 epoch 为一个完整周期
- 梯度裁剪：`max_norm=1.0`
- 每 100 epoch 保存一次 checkpoint（`checkpoints/checkpoint_epoch_{N}.pt`）
- 每 5 epoch 更新一次损失曲线图（3×3，包含归一化损失和原始 MSE）

**测试脚本**（位于 `tests/` 目录）：
- `tests/test_overfitting.py`：在单个 batch 上验证模型的过拟合能力（模型健全性检查）
- `tests/test_se3_equivariance.py`：验证 SE(3) 旋转等变性和平移不变性
- `tests/test_se3_mirror_symmetry.py`：验证镜像对称性（对映体 ECD 符号相反）

---

## 5. 评估与光谱生成

### 5.1 测试集评估（`scripts/04_evaluate.py`）

**功能**：加载训练好的模型 checkpoint，在完整测试集上计算各项指标，并批量生成光谱对比图。

**运行方式**：
```bash
# 使用默认 best_model.pt
python scripts/04_evaluate.py

# 指定 checkpoint
python scripts/04_evaluate.py --checkpoint checkpoints/checkpoint_epoch_1000.pt

# 只生成 50 张光谱对比图
python scripts/04_evaluate.py --num_spectra 50
```

脚本自动从 checkpoint 的 `config` 字段重建模型，无需手动填写超参数。

**输出指标**：
- 归一化总损失及各分量（E / μ_vel / m / R）
- 原始 MSE（Raw MSE）各分量
- R 符号预测准确率（R_sign_acc）

**输出文件**（默认保存至 `checkpoints/test_spectra/`）：
- `spectrum_mol_{id}.png`：每个分子的预测 vs 真实光谱对比图（100 张）
- `spectrum_stats.txt`：光谱差异统计（均值、中位数、标准差及逐分子误差）
- `evaluate.log`：完整评估日志

### 5.2 光谱生成原理

**目标**：从模型预测的离散激发态生成连续的 ECD 光谱曲线

**步骤 1：模型推理**
```python
pred = model(data)
E_pred = pred['E_pred']   # [20] 激发能 (eV)
R_pred = pred['R_pred']   # [20] 旋转强度 (10^-40 cgs)
```

**步骤 2：波长-能量转换**
```python
λ = [80, 81, ..., 450] nm   # 波长网格，步长 1 nm
E_grid = 1240 / λ            # eV
```

**步骤 3：高斯展宽**
```
Δε(E) = (1 / (2.296×10¹ × σ × √π)) × Σ_i E_i × R_cgs,i × exp[-(E - E_i)² / σ²]
```
- `R_cgs` 直接使用模型输出的 10^-40 cgs 值（`04_evaluate.py`、`05_generate_pred_spectrum.py`、`07_generate_real_spectrum.py` 三个脚本均统一采用此约定）
- `σ = 0.4 eV`：高斯展宽标准差
- `2.296×10¹`：归一化常数（对应 R 单位为 10^-40 cgs 时的数值）

**步骤 4：转换为摩尔椭圆度**
```
[θ] = Δε × 3298.2
```

### 5.3 光谱对比绘图（`scripts/08_plot_spectrum.py`）

**功能**：读取两个 CSV 文件（预测光谱和真实光谱），将其绘制在同一张图上。

**运行方式**：
```bash
python scripts/08_plot_spectrum.py \
    --pred_csv path/to/predicted.csv \
    --real_csv path/to/real.csv \
    --output_path path/to/output.png
```

**图像风格**：蓝色实线（Real）+ 红色虚线（Predicted），300 DPI，含基线、网格、图例。

---

## 6. 项目目录结构

```
PhysECD/
├── physecd/                        ← 核心库（模型、物理层、数据解析）
│   ├── models/
│   │   ├── physecd_model.py        完整模型（组合三大组件）
│   │   ├── se3_backbone.py         SE(3) 等变主干网络
│   │   ├── heads.py                多任务预测头
│   │   └── modules/                子模块（嵌入、径向基、交互块等）
│   ├── physics/
│   │   ├── aggregation.py          物理聚合层（无参数）
│   │   └── loss.py                 多任务损失函数（EMA 归一化）
│   └── data/
│       ├── parser.py               Gaussian .log 文件解析器
│       └── dataset_cmcds.py        CMCDS CSV 解析器
│
├── scripts/                        ← 按编号排列的执行脚本
│   ├── 01_prepare_data.py          数据集准备（解析 → PyG Data）
│   ├── 01_read_train_pt.py         数据检查工具
│   ├── 02_check_formula.py         物理公式严谨性验证（R = 6414.135151·μ·m/E）
│   ├── 02_generate_enantiomers.py  对映体生成（数据集 ×2）
│   ├── 03_train.py                 模型训练（训练 + 验证循环）
│   ├── 04_evaluate.py              测试集评估 + 批量光谱对比图
│   ├── 05_generate_pred_spectrum.py 单分子预测光谱生成
│   ├── 06_generate_pred_states.py  预测激发态输出（逐态 E/R 对比 CSV）
│   ├── 07_generate_real_spectrum.py 从 CSV 数据集生成真实光谱
│   └── 08_plot_spectrum.py         预测 vs 真实光谱对比绘图
│
├── tests/                          ← 架构健全性测试（改动模型时运行）
│   ├── test_overfitting.py         模型容量测试（单 batch 过拟合）
│   ├── test_se3_equivariance.py    SE(3) 旋转等变 + 平移不变性测试
│   └── test_se3_mirror_symmetry.py 手性镜像对称性测试
│
├── data/                           ← 数据集
│   ├── CMCDS_DATASET.csv
│   ├── CMCDS_DATASET_with_enantiomers.csv
│   ├── processed/                  原始数据集（无对映体）
│   └── processed_with_enantiomers/ 扩增数据集（含对映体，训练使用）
│
├── checkpoints/                    ← 训练产物
│   ├── best_model.pt               验证集最优模型
│   ├── checkpoint_epoch_{N}.pt     每 100 epoch 的定期 checkpoint
│   ├── loss_curves.png             训练过程损失曲线图
│   └── test_spectra/               04_evaluate.py 的光谱输出
│
├── documents/                      ← 项目文档
├── configs/                        ← 配置文件
└── CLAUDE.md                       ← Claude Code 项目指令
```
