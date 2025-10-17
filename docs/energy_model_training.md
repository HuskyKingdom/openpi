# Energy Model 训练指南

本指南介绍如何在 OpenPI 中训练和使用 Energy Model。

## 概述

Energy Model 是一个基于能量的模型，用于学习状态-动作对的能量函数。通过 InfoNCE 对比学习，模型学习为正确的状态-动作对分配低能量，为不匹配的对分配高能量。

## 架构

Energy Model 包含以下组件：

- **State/Context Encoder**: MLP ResNet，将状态表征映射到隐藏空间
- **Action Encoder**: MLP ResNet + Positional Encoding，处理动作序列
- **Cross Attention**: 让动作序列关注状态表征
- **Prediction Head**: MLP ResNet，预测能量值
- **Pooling**: 在动作序列维度上进行平均池化

## 训练模式

### 1. 仅监控 Energy Loss（默认）

在这种模式下，Energy Model 会被初始化和计算，但 energy loss 不参与训练，仅用于监控。

```python
# 训练配置
model = pi0_config.Pi0Config(
    use_energy_loss=False,  # Energy loss 仅用于监控
    energy_hidden=512,
    energy_heads=8,
    energy_layers=4,
)
```

运行训练：
```bash
python scripts/train.py --config pi0_libero
```

### 2. Energy Loss 参与训练

在这种模式下，energy loss 会加到总损失中，与 flow matching loss 一起优化。

```python
model = pi0_config.Pi0Config(
    use_energy_loss=True,  # Energy loss 参与训练
    energy_hidden=512,
    energy_heads=8,
    energy_layers=4,
)
```

**注意**: 默认权重是 `flow_loss + 0.1 * energy_loss`，可以在 `src/openpi/models/pi0.py` 第257行调整权重。

### 3. 仅训练 Energy Model（推荐用于预训练模型）

在这种模式下，Pi0 主模型的所有参数被冻结，只训练 Energy Model。这对于给已经训练好的策略模型添加能量模型非常有用。

```python
model = pi0_config.Pi0Config(
    use_energy_loss=True,  # 必须启用
    train_only_energy_model=True,  # 冻结除 energy_model 外的所有参数
    energy_hidden=512,
    energy_heads=8,
    energy_layers=4,
)
```

运行训练：
```bash
python scripts/train.py --config pi0_libero_energy_only
```

## 配置参数

### Pi0Config 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `energy_hidden` | int | 512 | Energy model 的隐藏层维度 |
| `energy_heads` | int | 8 | Multi-head attention 的头数 |
| `energy_layers` | int | 4 | MLP ResNet 的层数（未使用，保留扩展） |
| `use_energy_loss` | bool | False | 是否将 energy loss 加入训练损失 |
| `train_only_energy_model` | bool | False | 是否只训练 energy model，冻结其他参数 |

## 训练示例

### 完整训练（从头开始）

```bash
# 1. 训练基础 Pi0 模型
python scripts/train.py --config pi0_libero

# 2. 在训练好的模型上添加 energy model
python scripts/train.py --config pi0_libero_energy_only \
    --weight_loader.path ./checkpoints/pi0_libero/best/params
```

### 联合训练

如果想同时训练策略和 energy model：

```python
# 创建自定义配置
TrainConfig(
    name="pi0_libero_with_energy",
    model=pi0_config.Pi0Config(
        use_energy_loss=True,  # Energy loss 参与训练
        train_only_energy_model=False,  # 不冻结参数
    ),
    data=LeRobotLiberoDataConfig(...),
    weight_loader=weight_loaders.CheckpointWeightLoader(...),
    num_train_steps=30_000,
)
```

## 验证和测试

### 测试 Energy Model 实现

```bash
python test_energy_model.py
```

这将验证：
- Energy model 的前向传播
- InfoNCE loss 计算
- 梯度计算
- JIT 编译

### 测试参数冻结

```bash
python test_energy_freeze.py
```

这将验证：
- `train_only_energy_model=True` 时只有 energy_model 参数可训练
- `train_only_energy_model=False` 时所有参数可训练
- 参数路径正确

## 监控

训练过程中会打印 energy model 的统计信息：

```
Energy loss: 0.5234, E_pos: 1.2345, E_neg: 2.3456
```

其中：
- **Energy loss**: InfoNCE 对比损失
- **E_pos**: 正样本对（匹配的状态-动作对）的平均能量
- **E_neg**: 负样本对（不匹配的状态-动作对）的平均能量

**理想情况**: E_pos < E_neg，表示模型正确学习到匹配对应该有更低的能量。

## InfoNCE Loss 详解

Energy Model 使用 in-batch swap InfoNCE loss：

1. 对于 batch 中的每个样本 i，创建所有可能的（context_i, action_j）对
2. 正样本对：(context_i, action_i)
3. 负样本对：(context_i, action_j), j ≠ i
4. 使用交叉熵损失优化：`loss = -log(exp(-E_ii/τ) / Σ_j exp(-E_ij/τ))`

其中 τ 是温度参数（默认 0.5）。

## 常见问题

### Q: Energy loss 应该设置多大的权重？

A: 建议从小权重开始（如 0.1），观察训练稳定性。权重在 `src/openpi/models/pi0.py` 第257行。

### Q: 为什么只使用前7维动作？

A: Pi0 内部使用32维动作空间以支持多种机器人。对于 LIBERO（7维），只有前7维有意义，后25维是填充。可以在 `src/openpi/models/pi0.py` 第237行调整。

### Q: 可以单独保存 energy model 吗？

A: 可以。使用以下代码：

```python
import flax.nnx as nnx

# 获取 energy_model 的参数
energy_params = nnx.state(model.energy_model, nnx.Param)

# 保存
with open("energy_model.pkl", "wb") as f:
    pickle.dump(energy_params, f)
```

### Q: Energy model 的计算开销如何？

A: Energy model 相对较小（~2M 参数），计算开销主要在 cross attention 上。在训练时增加约 10-15% 的时间。

## 技术细节

### Mask 语义

- **Pi0 中**: `True` = 有效输入, `False` = padding
- **JAX Attention**: `True` = padding, `False` = 有效输入
- **解决方案**: 传给 energy model 前自动反转 mask

### 梯度流

Energy loss 的梯度会回传到：
- Energy model 的所有参数（state_linear, action_linear, cross_attention, prediction_head）
- 如果 `train_only_energy_model=False`，还会回传到 Pi0 的 prefix_out（即 LLM 的输出）

### JIT 编译

Energy model 完全支持 JAX 的 JIT 编译，不会触发 tracer 错误。这是将 PyTorch 版本转换为 JAX 的主要动机。

## 参考

- [Flow Matching 论文](https://arxiv.org/abs/2210.02747)
- [InfoNCE Loss](https://arxiv.org/abs/1807.03748)
- [Energy-Based Models](https://arxiv.org/abs/2101.03288)

