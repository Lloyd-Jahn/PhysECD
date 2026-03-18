"""
PhysECD - 用于 ECD 光谱预测的物理驱动 SE(3) 框架
====================================================================

一个深度学习框架，结合了 SE(3) 等变神经网络与基于物理的聚合层，
用于从分子 3D 结构预测电子圆二色性 (ECD) 光谱。

模块:
- data: 数据解析与数据集工具
- models: 神经网络架构
- physics: 基于物理的聚合与渲染层
"""

__version__ = "0.1.0"

from .models import PhysECDModel, SE3Backbone, MultiTaskHeads
from .physics import PhysicsAggregation, PhysECDLoss

__all__ = [
    'PhysECDModel',
    'SE3Backbone',
    'MultiTaskHeads',
    'PhysicsAggregation',
    'PhysECDLoss'
]