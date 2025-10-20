# Energy-Based Action Correction

## 概述

Energy-Based Action Correction 是一种测试时优化技术，通过在能量景观上进行梯度下降来改善策略预测的动作质量。

## 原理

### 能量模型

Energy Model `E_φ(s, a)` 学习为合理的状态-动作对分配低能量，为不合理的对分配高能量：

- **正样本对** (s_i, a_i)：来自专家演示，应有**低能量**
- **负样本对** (s_i, a_j), i≠j：不匹配的对，应有**高能量**

### 训练目标

使用 InfoNCE 对比学习损失：

```
L = -log(exp(-E_ii/τ) / Σ_j exp(-E_ij/τ))
```

其中 τ 是温度参数。

### 测试时校正

在推理时，给定策略预测的动作 `a_pred`，通过梯度下降优化：

```
a_corrected = a_pred - α * ∇_a E(s, a_pred)
```

其中 α 是步长，∇_a E 是能量对动作的梯度。

## 使用方法

### 1. 训练 Energy Model

首先训练一个 energy model：

```bash
uv run scripts/train.py pi05_libero_energy --exp-name=energy_v1 --overwrite
```

关键配置：
```python
model=pi0_config.Pi0Config(
    use_energy_loss=True,           # 启用 energy loss
    train_only_energy_model=True,   # 冻结策略参数
    energy_act_dim=7,                # LIBERO 动作维度
    energy_hidden=512,               # Energy model 隐藏层维度
    energy_heads=8,                  # Attention heads
)
```

### 2. 在代码中使用 Energy Correction

#### 方法 A：使用内置的 sample_actions_with_energy_correction

```python
import jax
from openpi.training import config as _config
from openpi.models import model as _model

# 加载训练好的模型（包含 energy model）
config = _config.get_config("pi05_libero_energy")
checkpoint_dir = "./checkpoints/pi05_libero_energy/energy_v1"
params = _model.restore_params(checkpoint_dir / "params")
model = config.model.load(params)
model.eval()

# 创建观测
obs = ...  # 你的观测数据

# 推理时应用 energy correction
rng = jax.random.key(0)
actions = model.sample_actions_with_energy_correction(
    rng,
    obs,
    num_steps=10,                    # Flow matching 步数
    energy_correction_steps=3,       # Energy 校正迭代次数
    energy_alpha=0.1,                # 校正步长
    energy_clip_frac=0.2,            # 最大校正幅度
    correct_first_only=False,        # 是否只校正第一个动作
)
```

#### 方法 B：手动调用 correction 函数

如果需要更灵活的控制：

```python
from openpi.models.energy_correction import multi_step_energy_correction

# 1. 获取基础动作预测
actions_base = model.sample_actions(rng, obs, num_steps=10)

# 2. 获取 context representation
observation = _model.preprocess_observation(None, obs, train=False)
prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
(prefix_out,), _ = model.PaliGemma.llm(
    [prefix_tokens, None], 
    mask=prefix_attn_mask, 
    positions=positions,
    adarms_cond=[None, None],
)

# 3. 应用 energy correction
inverted_prefix_mask = ~prefix_mask  # Energy model expects True=padding
actions_for_correction = actions_base[:, :, :7]  # LIBERO uses 7-dim

corrected_actions = multi_step_energy_correction(
    model.energy_model,
    prefix_out,
    actions_for_correction,
    pad_mask=inverted_prefix_mask,
    num_steps=3,
    alpha=0.1,
    clip_frac=0.2,
    train=False,
)

# 4. 组合回完整的动作向量
final_actions = actions_base.at[:, :, :7].set(corrected_actions)
```

### 3. 在 LIBERO 评估中集成

修改 `examples/libero/main.py` 来使用 energy correction：

```python
# 在 eval_libero 函数中，替换动作采样部分：

# 原来的代码：
action_chunk = client.infer(element)["actions"]

# 使用 energy correction：
if args.use_energy_correction:
    # 假设 client 返回了带有 energy correction 的动作
    # 或者在服务器端启用 correction
    action_chunk = client.infer(element)["actions"]
else:
    action_chunk = client.infer(element)["actions"]
```

如果你想在客户端本地执行 correction（不通过服务器），可以直接加载模型：

```python
# 在脚本开头加载模型
model = config.model.load(params)
model.eval()

# 在推理循环中
obs_for_model = ...  # 转换 element 为模型格式
actions_corrected = model.sample_actions_with_energy_correction(
    rng, obs_for_model, 
    energy_correction_steps=3,
    energy_alpha=0.1,
)
```

## 参数调优指南

### energy_correction_steps (迭代次数)

- **1-3 步**：快速，适合实时应用（< 50ms 额外延迟）
- **5-10 步**：更好的校正，但较慢（~100-200ms）
- **0 步**：禁用 correction（baseline）

### energy_alpha (步长)

- **0.05-0.10**：保守，小幅度校正（推荐起点）
- **0.10-0.20**：中等幅度
- **0.20-0.50**：激进，可能导致不稳定

**建议**：从 0.1 开始，根据评估结果调整

### energy_clip_frac (裁剪比例)

限制每步的最大变化为原始动作范数的比例：

- **0.1**：非常保守
- **0.2**：适中（推荐）
- **0.5**：允许较大变化

### correct_first_only (是否只校正第一个动作)

- **False**：校正整个动作序列（推荐）
- **True**：只校正 chunk 的第一个动作，其余保持不变
  - 优点：更快，更稳定
  - 缺点：长期规划可能不够优化

## 性能影响

### 计算开销

Energy correction 增加的计算时间（相对于基础推理）：

| 迭代次数 | 额外时间 | 总推理时间增加 |
|---------|---------|---------------|
| 1 步    | ~10ms   | ~5%           |
| 3 步    | ~30ms   | ~15%          |
| 5 步    | ~50ms   | ~25%          |

**注意**：实际时间取决于硬件（GPU）和 batch size。

### 内存使用

Energy correction 使用额外内存来：
- 存储梯度：~动作大小
- Energy model 前向传播：小（energy model 只有 ~2M 参数）

增加的内存使用通常 < 10%。

## 监控和调试

### 检查 Energy Reduction

在应用 correction 后，energy 应该下降：

```python
# 计算初始能量
E_before = model.energy_model(h, actions_base, pad_mask, train=False)

# 应用 correction
actions_corrected = multi_step_energy_correction(...)

# 计算校正后能量
E_after = model.energy_model(h, actions_corrected, pad_mask, train=False)

# 检查能量是否下降
print(f"Energy before: {E_before.mean():.4f}")
print(f"Energy after:  {E_after.mean():.4f}")
print(f"Reduction:     {E_before.mean() - E_after.mean():.4f}")
```

**期望**：`E_after < E_before`（能量下降）

### 可视化能量景观

```python
# 扫描不同的 alpha 值
alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
energies = []

for alpha in alphas:
    corrected = one_step_energy_correction(
        model.energy_model, h, actions, alpha=alpha
    )
    E = model.energy_model(h, corrected, pad_mask, train=False)
    energies.append(E.mean())

# 绘制能量 vs alpha
import matplotlib.pyplot as plt
plt.plot(alphas, energies)
plt.xlabel('Alpha (step size)')
plt.ylabel('Energy after correction')
plt.title('Energy Landscape')
plt.show()
```

## 常见问题

### Q: Energy correction 没有降低能量？

**可能原因**：
1. Energy model 没有训练好
2. Alpha 太大或太小
3. Energy model 和策略不匹配

**解决方案**：
- 确保 energy model 已训练至收敛（E_pos < E_neg）
- 尝试不同的 alpha 值（0.05-0.2）
- 检查训练数据和测试场景是否一致

### Q: Correction 后动作变化太大？

**解决方案**：
- 减小 `alpha`（如 0.05）
- 减小 `clip_frac`（如 0.1）
- 减少 `energy_correction_steps`

### Q: 能否与 flow matching 联合优化？

可以，但需要修改 sampling 过程，在每个 flow step 后应用 energy correction。这更复杂但可能更有效。

### Q: Correction 的计算开销能否减小？

可以通过以下方式：
- 使用 `correct_first_only=True`（只校正第一个动作）
- 减少 energy model 大小（减小 `energy_hidden`）
- 使用更少的 correction steps

## 技术细节

### 梯度计算

使用 JAX 的自动微分：

```python
def energy_fn(actions_var):
    return jnp.sum(energy_model(h, actions_var, pad_mask))

grad_actions = jax.grad(energy_fn)(actions)
```

### 步长裁剪

防止过大的校正：

```python
step = alpha * grad_actions
step_norm = ||step||
max_step = clip_frac * ||actions||
if step_norm > max_step:
    step = step * (max_step / step_norm)
```

### JIT 编译

所有 correction 函数都支持 JIT 编译：

```python
jitted_correction = jax.jit(
    lambda h, a, m: one_step_energy_correction(
        energy_model, h, a, m, alpha=0.1
    )
)
```

## 实验建议

### Baseline 对比

建议对比以下设置：

1. **No correction**：baseline 性能
2. **1-step correction, α=0.1**：最小开销
3. **3-step correction, α=0.1**：平衡性能和开销
4. **5-step correction, α=0.1**：最大校正质量

### 评估指标

- **成功率**：任务完成率
- **执行效率**：完成步数
- **能量统计**：E_before vs E_after
- **动作变化**：||a_corrected - a_base||

## 参考

- Energy-Based Models: [LeCun et al., 2006](https://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
- InfoNCE Loss: [Oord et al., 2018](https://arxiv.org/abs/1807.03748)
- Test-Time Optimization: [Sun et al., 2019](https://arxiv.org/abs/1912.08570)

