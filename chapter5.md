# 第5章：深度学习优化

深度学习的成功很大程度上依赖于优化算法的进步。本章将从优化理论的角度深入探讨深度神经网络的训练方法，包括随机梯度下降的各种变体、自适应学习率算法、归一化技术以及二阶优化方法。我们将特别关注这些方法背后的统计学原理，以及它们在现代大规模模型训练中的实际应用。

## 5.1 随机梯度下降（SGD）及其动量变体

### 5.1.1 从批量梯度到随机梯度

在深度学习中，我们通常面对的优化问题是：

$$\min_{\theta} \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(f_\theta(\mathbf{x}_i), y_i)$$

其中 $N$ 是训练样本数，$\ell$ 是损失函数，$f_\theta$ 是参数为 $\theta$ 的神经网络。

**批量梯度下降（BGD）** 使用全部数据计算梯度：
$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

但当 $N$ 很大时，计算成本过高。

**随机梯度下降（SGD）** 每次只使用一个样本：
$$\theta_{t+1} = \theta_t - \eta \nabla \ell(f_{\theta_t}(\mathbf{x}_i), y_i)$$

**小批量SGD（Mini-batch SGD）** 是实践中的标准做法：
$$\theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla \ell(f_{\theta_t}(\mathbf{x}_i), y_i)$$

其中 $\mathcal{B}_t$ 是大小为 $B$ 的小批量。

### 5.1.2 SGD的统计性质

从统计角度看，SGD的梯度估计是无偏的：
$$\mathbb{E}[\nabla \ell(f_{\theta}(\mathbf{x}_i), y_i)] = \nabla \mathcal{L}(\theta)$$

但存在方差：
$$\text{Var}[\nabla \ell(f_{\theta}(\mathbf{x}_i), y_i)] = \sigma^2$$

小批量可以降低方差：
$$\text{Var}\left[\frac{1}{B} \sum_{i \in \mathcal{B}} \nabla \ell_i\right] = \frac{\sigma^2}{B}$$

**经验法则**：
- 批量大小通常选择2的幂次（32, 64, 128, 256）以利用硬件加速
- 增大批量大小时，学习率可以线性增加（线性缩放规则）
- 批量大小存在临界值，超过后收敛性能不再改善

### 5.1.3 动量方法

**经典动量（Momentum）**：
$$\begin{aligned}
\mathbf{v}_{t+1} &= \beta \mathbf{v}_t - \eta \nabla \mathcal{L}(\theta_t) \\
\theta_{t+1} &= \theta_t + \mathbf{v}_{t+1}
\end{aligned}$$

其中 $\beta \in [0,1)$ 是动量系数，典型值为0.9。

动量方法可以看作是指数移动平均：
$$\mathbf{v}_t = -\eta \sum_{i=0}^{t} \beta^{t-i} \nabla \mathcal{L}(\theta_i)$$

```
无动量：  ↗↘↗↘  (振荡)
有动量：  →→→→  (平滑)
```

**Nesterov加速梯度（NAG）**：
$$\begin{aligned}
\mathbf{v}_{t+1} &= \beta \mathbf{v}_t - \eta \nabla \mathcal{L}(\theta_t + \beta \mathbf{v}_t) \\
\theta_{t+1} &= \theta_t + \mathbf{v}_{t+1}
\end{aligned}$$

NAG在计算梯度前先"向前看"，具有更好的理论收敛速度。

### 5.1.4 学习率调度

学习率调度对深度学习至关重要：

**阶梯衰减（Step Decay）**：
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$
其中 $\gamma < 1$ 是衰减因子，$s$ 是步长。

**余弦退火（Cosine Annealing）**：
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**线性预热（Linear Warmup）**：
$$\eta_t = \begin{cases}
\frac{t}{T_{warmup}} \eta_{target} & t < T_{warmup} \\
\eta_{target} & t \geq T_{warmup}
\end{cases}$$

**经验法则**：
- 训练初期使用预热，避免梯度爆炸
- 大批量训练需要更长的预热期
- 余弦退火通常优于阶梯衰减

## 5.2 Adam与自适应学习率方法

### 5.2.1 AdaGrad：自适应梯度算法

AdaGrad为每个参数维持独立的学习率：
$$\begin{aligned}
\mathbf{g}_t &= \nabla \mathcal{L}(\theta_t) \\
\mathbf{G}_t &= \mathbf{G}_{t-1} + \mathbf{g}_t^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \odot \mathbf{g}_t
\end{aligned}$$

其中 $\odot$ 表示逐元素乘法，$\epsilon$ 是小常数（如 $10^{-8}$）防止除零。

问题：$\mathbf{G}_t$ 单调递增，学习率最终趋于零。

### 5.2.2 RMSprop：指数移动平均

RMSprop解决了AdaGrad学习率递减问题：
$$\begin{aligned}
\mathbf{g}_t &= \nabla \mathcal{L}(\theta_t) \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) \mathbf{g}_t^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \mathbf{g}_t
\end{aligned}$$

典型的 $\beta_2 = 0.999$。

### 5.2.3 Adam：自适应矩估计

Adam结合了动量和RMSprop的优点：
$$\begin{aligned}
\mathbf{g}_t &= \nabla \mathcal{L}(\theta_t) \\
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \mathbf{g}_t & \text{(一阶矩估计)} \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) \mathbf{g}_t^2 & \text{(二阶矩估计)} \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1-\beta_1^t} & \text{(偏差修正)} \\
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1-\beta_2^t} & \text{(偏差修正)} \\
\theta_{t+1} &= \theta_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{aligned}$$

**默认超参数**：
- $\beta_1 = 0.9$（动量系数）
- $\beta_2 = 0.999$（二阶矩衰减率）
- $\eta = 0.001$（学习率）
- $\epsilon = 10^{-8}$

### 5.2.4 Adam的变体

**AdamW（权重衰减解耦）**：
$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \theta_t\right)$$

将L2正则化从梯度计算中分离，改善泛化性能。

**RAdam（矫正Adam）**：
自动调整自适应学习率的方差，解决训练初期的不稳定问题。

**经验法则**：
- Adam通常是深度学习的默认选择
- 对于视觉任务，SGD+动量可能获得更好的泛化
- 对于NLP和Transformer，Adam/AdamW效果更好
- 学习率通常需要根据模型大小调整

## 5.3 批归一化与层归一化

### 5.3.1 内部协变量偏移问题

深度网络训练中，每层输入的分布随着前层参数更新而改变，这种现象称为内部协变量偏移（Internal Covariate Shift）。

### 5.3.2 批归一化（Batch Normalization）

批归一化通过标准化激活值来稳定训练：

**训练时**：
$$\begin{aligned}
\mu_B &= \frac{1}{B} \sum_{i=1}^{B} \mathbf{x}_i \\
\sigma_B^2 &= \frac{1}{B} \sum_{i=1}^{B} (\mathbf{x}_i - \mu_B)^2 \\
\hat{\mathbf{x}}_i &= \frac{\mathbf{x}_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\
\mathbf{y}_i &= \gamma \hat{\mathbf{x}}_i + \beta
\end{aligned}$$

其中 $\gamma$ 和 $\beta$ 是可学习的缩放和偏移参数。

**推理时**：
使用训练时的移动平均统计量：
$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu_{running}}{\sqrt{\sigma_{running}^2 + \epsilon}}$$

### 5.3.3 层归一化（Layer Normalization）

层归一化在特征维度上进行标准化，不依赖批量大小：

$$\begin{aligned}
\mu_l &= \frac{1}{H} \sum_{i=1}^{H} x_i \\
\sigma_l^2 &= \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu_l)^2 \\
\hat{x}_i &= \frac{x_i - \mu_l}{\sqrt{\sigma_l^2 + \epsilon}}
\end{aligned}$$

其中 $H$ 是隐藏层维度。

**批归一化 vs 层归一化**：
```
批归一化：跨批量样本归一化
  样本1: [x11, x12, x13]  ↓
  样本2: [x21, x22, x23]  ↓ 
  样本3: [x31, x32, x33]  ↓
         归一化每一列

层归一化：跨特征维度归一化
  样本1: [x11, x12, x13] → 归一化
  样本2: [x21, x22, x23] → 归一化
  样本3: [x31, x32, x33] → 归一化
```

### 5.3.4 其他归一化技术

**组归一化（Group Normalization）**：
将通道分组，在每组内进行归一化，适用于小批量场景。

**实例归一化（Instance Normalization）**：
每个样本独立归一化，常用于风格迁移。

**RMSNorm**：
简化的层归一化，只使用二阶矩：
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \gamma$$
其中 $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{H}\sum_{i=1}^{H} x_i^2}$

**经验法则**：
- CNN通常使用批归一化
- RNN和Transformer使用层归一化
- 批量大小小于32时考虑组归一化
- 大模型训练倾向于使用RMSNorm（计算效率更高）

## 5.4 二阶优化方法

### 5.4.1 牛顿法与拟牛顿法

**牛顿法**使用Hessian矩阵（二阶导数）：
$$\theta_{t+1} = \theta_t - \eta \mathbf{H}^{-1} \nabla \mathcal{L}(\theta_t)$$

其中 $\mathbf{H} = \nabla^2 \mathcal{L}(\theta)$ 是Hessian矩阵。

问题：
- 计算Hessian需要 $O(n^2)$ 存储和 $O(n^3)$ 计算
- Hessian可能不正定，导致更新方向错误

### 5.4.2 L-BFGS

L-BFGS（Limited-memory BFGS）通过低秩近似避免存储完整Hessian：

使用最近 $m$ 步的梯度信息构造Hessian逆的近似：
$$\mathbf{s}_k = \theta_{k+1} - \theta_k, \quad \mathbf{y}_k = \nabla \mathcal{L}_{k+1} - \nabla \mathcal{L}_k$$

通过递归计算得到搜索方向，内存需求为 $O(mn)$。

### 5.4.3 自然梯度下降

自然梯度考虑参数空间的几何结构：
$$\theta_{t+1} = \theta_t - \eta \mathbf{F}^{-1} \nabla \mathcal{L}(\theta_t)$$

其中 $\mathbf{F}$ 是Fisher信息矩阵：
$$\mathbf{F} = \mathbb{E}_{p(x|\theta)}\left[\nabla \log p(x|\theta) \nabla \log p(x|\theta)^T\right]$$

### 5.4.4 K-FAC（Kronecker分解近似）

K-FAC通过Kronecker积近似Fisher矩阵：
$$\mathbf{F} \approx \mathbf{A} \otimes \mathbf{G}$$

其中 $\mathbf{A}$ 是输入协方差，$\mathbf{G}$ 是梯度协方差。

优点：
- 计算效率高于完整Fisher矩阵
- 可以并行计算
- 在某些任务上收敛更快

### 5.4.5 Shampoo优化器

Shampoo是一种实用的二阶优化器，通过预条件矩阵加速收敛：
$$\theta_{t+1} = \theta_t - \eta \mathbf{H}_t^{-1/4} \nabla \mathcal{L}(\theta_t) \mathbf{G}_t^{-1/4}$$

其中 $\mathbf{H}_t$ 和 $\mathbf{G}_t$ 分别是左右预条件矩阵。

**经验法则**：
- 二阶方法在小到中等规模问题上效果好
- 大规模深度学习通常使用一阶方法（计算效率）
- L-BFGS适合批量训练，不适合随机优化
- K-FAC和Shampoo在某些场景可以加速训练2-3倍

## 5.5 历史人物：Diederik P. Kingma与Adam优化器的革命

Diederik P. Kingma是现代深度学习优化算法的关键贡献者。2014年，他与Jimmy Ba共同提出了Adam优化器，这篇论文成为机器学习领域引用最多的论文之一，彻底改变了深度学习的训练方式。

### 学术轨迹

Kingma在阿姆斯特丹大学获得博士学位，师从Max Welling教授。他的研究兴趣横跨变分推断、生成模型和优化理论。除了Adam，他还是变分自编码器（VAE）的共同发明者，这两项工作都对深度学习产生了深远影响。

### Adam的诞生背景

2014年前，深度学习社区主要使用SGD及其动量变体。虽然AdaGrad和RMSprop等自适应方法已经出现，但都存在各自的问题：
- AdaGrad学习率单调递减
- RMSprop缺乏理论保证
- 不同任务需要不同的优化器

Kingma意识到需要一个通用、鲁棒的优化器，能够自适应地调整学习率，同时保持动量的优势。

### 核心创新

Adam的关键创新在于：
1. **结合一阶和二阶矩估计**：同时追踪梯度的均值和方差
2. **偏差修正**：解决初始化偏差问题
3. **逐参数自适应**：为每个参数维持独立的学习率
4. **理论保证**：提供了收敛性证明

### 影响与争议

Adam迅速成为深度学习的默认优化器，特别是在：
- 自然语言处理
- 生成对抗网络
- 变分自编码器
- Transformer模型

但也存在争议：
- Wilson等人(2017)指出Adam可能导致泛化性能下降
- 导致了AdamW等改进版本的出现

### 后续贡献

Kingma继续在优化和生成模型领域做出贡献：
- Glow：基于流的生成模型
- 改进的变分推断方法
- 扩散模型的理论基础

他的工作体现了理论与实践的完美结合，不仅提出了实用的算法，还提供了严格的理论分析。Adam优化器的成功证明了在深度学习中，好的优化算法可以极大地推动整个领域的进步。

## 5.6 现代连接：LLM训练中的梯度检查点与ZeRO优化

### 5.6.1 梯度检查点（Gradient Checkpointing）

训练大模型的主要挑战是内存限制。标准反向传播需要存储所有中间激活值：

**内存需求**：
- 前向传播：$O(n \cdot l)$（$n$是批量大小，$l$是层数）
- 反向传播：需要所有中间激活值

**梯度检查点策略**：
1. 前向传播时只保存部分检查点
2. 反向传播时重新计算未保存的激活值
3. 时间换空间：增加33%计算，减少$O(\sqrt{l})$内存

```
标准BP：  [A1][A2][A3][A4][A5][A6][A7][A8] (保存所有)
检查点：  [A1]    [A3]    [A5]    [A7]     (保存部分)
         重算A2  重算A4  重算A6  重算A8
```

### 5.6.2 ZeRO优化器（Zero Redundancy Optimizer）

ZeRO通过分片技术减少内存冗余：

**ZeRO-1：优化器状态分片**
- Adam需要存储：参数、梯度、动量、方差
- 内存需求：$4 \times$ 模型大小
- ZeRO-1：将优化器状态分布到多个GPU

**ZeRO-2：梯度分片**
- 每个GPU只存储部分梯度
- 通过all-reduce聚合梯度

**ZeRO-3：参数分片**
- 模型参数也分布存储
- 需要时通过通信获取

**内存节省**（以GPT-3 175B为例）：
```
标准数据并行：每GPU需要 1.4TB 内存
ZeRO-1：      每GPU需要 350GB
ZeRO-2：      每GPU需要 175GB  
ZeRO-3：      每GPU需要 20GB
```

### 5.6.3 混合精度训练

使用FP16/BF16代替FP32：

**核心技术**：
1. **主权重副本**：FP32主权重，FP16计算
2. **损失缩放**：防止梯度下溢
3. **动态损失缩放**：自适应调整缩放因子

```python
# 伪代码
loss = compute_loss(model_fp16, data)
scaled_loss = loss * scale_factor
scaled_gradients = backward(scaled_loss)
gradients = scaled_gradients / scale_factor
optimizer.step(gradients, master_weights_fp32)
```

### 5.6.4 大模型训练的优化策略组合

**典型配置**（训练100B+参数模型）：
1. **优化器**：AdamW + 梯度裁剪
2. **学习率**：余弦衰减 + 线性预热
3. **内存优化**：ZeRO-3 + 梯度检查点
4. **精度**：BF16混合精度
5. **并行策略**：
   - 数据并行（DP）
   - 张量并行（TP）
   - 流水线并行（PP）
   - 序列并行（SP）

**3D并行示例**：
```
模型分片：
  GPU集群 (8×8×8 = 512 GPUs)
  ├── PP=8 (流水线8段)
  ├── TP=8 (张量并行8路)
  └── DP=8 (数据并行8份)
```

### 5.6.5 通信优化

**Ring-AllReduce**：
环形拓扑减少通信开销，时间复杂度 $O(n)$ → $O(\log n)$

**梯度累积**：
```python
for micro_batch in mini_batch:
    loss = forward(micro_batch) / accumulation_steps
    backward(loss)
if step % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**经验法则**：
- 梯度检查点在层数>50时效果明显
- ZeRO-2通常是最佳平衡点
- BF16比FP16更稳定（动态范围更大）
- 通信成本随GPU数量增加，需要权衡并行度

## 本章小结

本章系统介绍了深度学习优化的核心方法和实践技巧：

### 关键概念

1. **随机梯度下降及其变体**
   - SGD通过小批量降低计算成本
   - 动量方法平滑优化轨迹：$\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \eta \nabla \mathcal{L}$
   - 学习率调度对收敛至关重要

2. **自适应学习率方法**
   - Adam结合动量和自适应学习率：$\theta_{t+1} = \theta_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$
   - AdamW将权重衰减从梯度中解耦
   - 默认超参数：$\beta_1=0.9$, $\beta_2=0.999$, $\eta=0.001$

3. **归一化技术**
   - 批归一化：跨批量样本归一化，适用于CNN
   - 层归一化：跨特征维度归一化，适用于Transformer
   - RMSNorm：简化版层归一化，计算效率更高

4. **二阶优化方法**
   - 利用Hessian信息加速收敛
   - L-BFGS、K-FAC等方法在特定场景有效
   - 计算成本限制了在大规模深度学习中的应用

5. **大规模训练技术**
   - 梯度检查点：时间换空间
   - ZeRO优化：分片减少内存冗余
   - 混合精度：FP16/BF16加速计算
   - 3D并行：数据、张量、流水线并行结合

### 实用建议

| 场景 | 推荐配置 |
|------|---------|
| 小规模实验 | Adam + 固定学习率 |
| CNN训练 | SGD+动量 + 批归一化 + 阶梯衰减 |
| Transformer | AdamW + 层归一化 + 余弦退火 + 预热 |
| 大模型训练 | AdamW + RMSNorm + ZeRO-2 + BF16 |
| 微调 | 较小学习率 + AdamW + 梯度裁剪 |

### 核心公式汇总

- **SGD with Momentum**: $\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \eta \nabla \mathcal{L}$
- **Adam更新规则**: 
  - $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \mathbf{g}_t$
  - $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) \mathbf{g}_t^2$
  - $\theta_{t+1} = \theta_t - \eta \frac{\mathbf{m}_t/(1-\beta_1^t)}{\sqrt{\mathbf{v}_t/(1-\beta_2^t)} + \epsilon}$
- **批归一化**: $\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
- **层归一化**: $\hat{x}_i = \frac{x_i - \mu_l}{\sqrt{\sigma_l^2 + \epsilon}}$

优化是深度学习成功的关键。选择合适的优化器和超参数，结合现代加速技术，可以显著提升模型训练效率和最终性能。

## 练习题

### 基础题

**练习5.1** 考虑一个二次函数 $f(x) = \frac{1}{2}x^TAx - b^Tx$，其中 $A$ 是正定矩阵。
- (a) 推导梯度下降的更新公式
- (b) 证明当学习率 $\eta < \frac{2}{\lambda_{max}(A)}$ 时，梯度下降收敛
- (c) 说明动量如何影响收敛速度

*提示：考虑特征值分解 $A = Q\Lambda Q^T$*

<details>
<summary>答案</summary>

(a) 梯度为 $\nabla f(x) = Ax - b$，更新公式：$x_{t+1} = x_t - \eta(Ax_t - b)$

(b) 令 $e_t = x_t - x^*$ 为误差，则：
   $e_{t+1} = (I - \eta A)e_t$
   收敛条件：$\rho(I - \eta A) < 1$
   即 $|1 - \eta \lambda_i| < 1$ 对所有特征值 $\lambda_i$
   因此需要 $\eta < \frac{2}{\lambda_{max}}$

(c) 动量方法的收敛率约为 $\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)$，其中 $\kappa = \frac{\lambda_{max}}{\lambda_{min}}$ 是条件数。无动量时收敛率为 $\frac{\kappa-1}{\kappa+1}$，动量显著改善了收敛速度。
</details>

**练习5.2** 比较SGD和Adam在以下场景的表现：
- (a) 稀疏梯度（如词嵌入训练）
- (b) 非平稳目标（如GAN训练）
- (c) 需要精确收敛到最优解

*提示：考虑自适应学习率的影响*

<details>
<summary>答案</summary>

(a) 稀疏梯度：Adam更好，因为它为每个参数维持独立的学习率，稀疏更新的参数能保持较大学习率

(b) 非平稳目标：Adam更好，自适应学习率能够快速适应目标函数的变化

(c) 精确收敛：SGD更好，Adam的自适应学习率可能导致在最优点附近振荡，而SGD配合学习率衰减能够精确收敛
</details>

**练习5.3** 实现批归一化的前向和反向传播。给定输入 $\mathbf{x} \in \mathbb{R}^{B \times D}$（批量大小$B$，特征维度$D$），计算：
- (a) 前向传播输出
- (b) 对输入的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$
- (c) 对参数 $\gamma, \beta$ 的梯度

*提示：使用链式法则，注意 $\mu$ 和 $\sigma^2$ 也依赖于 $\mathbf{x}$*

<details>
<summary>答案</summary>

(a) 前向传播：
   - $\mu = \frac{1}{B}\sum_i x_i$
   - $\sigma^2 = \frac{1}{B}\sum_i (x_i - \mu)^2$
   - $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
   - $y_i = \gamma \hat{x}_i + \beta$

(b) 反向传播（给定 $\frac{\partial \mathcal{L}}{\partial y}$）：
   - $\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \gamma$
   - $\frac{\partial \mathcal{L}}{\partial \sigma^2} = \sum_i \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot (x_i - \mu) \cdot (-\frac{1}{2})(\sigma^2 + \epsilon)^{-3/2}$
   - $\frac{\partial \mathcal{L}}{\partial \mu} = \sum_i \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot (-\frac{1}{\sqrt{\sigma^2 + \epsilon}})$
   - $\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial \mathcal{L}}{\partial \sigma^2} \cdot \frac{2(x_i - \mu)}{B} + \frac{\partial \mathcal{L}}{\partial \mu} \cdot \frac{1}{B}$

(c) 参数梯度：
   - $\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_i \frac{\partial \mathcal{L}}{\partial y_i} \cdot \hat{x}_i$
   - $\frac{\partial \mathcal{L}}{\partial \beta} = \sum_i \frac{\partial \mathcal{L}}{\partial y_i}$
</details>

**练习5.4** 分析不同批量大小对优化的影响：
- (a) 批量大小如何影响梯度估计的方差？
- (b) 为什么大批量训练可能导致泛化性能下降？
- (c) 线性缩放规则的理论依据是什么？

*提示：考虑梯度噪声的作用*

<details>
<summary>答案</summary>

(a) 梯度方差与批量大小成反比：$\text{Var}[\nabla_B] = \frac{\sigma^2}{B}$，其中 $B$ 是批量大小

(b) 大批量训练的问题：
   - 梯度噪声减少，可能陷入尖锐最小值（泛化性差）
   - 探索能力下降，难以跳出局部最优
   - 每个epoch的参数更新次数减少

(c) 线性缩放规则：批量大小增加 $k$ 倍时，学习率也增加 $k$ 倍
   理论依据：保持每次更新的期望变化量不变
   $k$ 个小批量的梯度和 ≈ $k$ 倍的大批量梯度
</details>

### 挑战题

**练习5.5** 推导Adam优化器的收敛界。假设：
- 梯度有界：$\|\nabla f(x)\| \leq G$
- 目标函数 $L$-smooth：$\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$

证明Adam的收敛率为 $O(1/\sqrt{T})$。

*提示：使用 regret bound 分析*

<details>
<summary>答案</summary>

定义regret：$R_T = \sum_{t=1}^T f(x_t) - f(x^*)$

关键步骤：
1. 建立递归关系：利用L-smooth性质
   $f(x_{t+1}) \leq f(x_t) + \langle \nabla f(x_t), x_{t+1} - x_t \rangle + \frac{L}{2}\|x_{t+1} - x_t\|^2$

2. 代入Adam更新规则，使用偏差修正后的估计

3. 利用自适应学习率的性质：
   $\sum_{t=1}^T \frac{\|g_t\|^2}{\sqrt{v_t}} \leq 2G\sqrt{T\sum_{t=1}^T \|g_t\|^2}$

4. 最终得到：$R_T \leq O(\sqrt{T})$，即平均收敛率 $O(1/\sqrt{T})$
</details>

**练习5.6** 设计一个自适应的学习率调度策略，要求：
- 自动检测损失平台期
- 在平台期降低学习率
- 避免过早降低学习率

描述你的算法并分析其优缺点。

*提示：可以使用移动平均检测平台*

<details>
<summary>答案</summary>

**自适应学习率调度算法**：

```
初始化：patience = 10, factor = 0.5, threshold = 0.01
best_loss = inf, wait = 0

for epoch in training:
    current_loss = evaluate()
    relative_improvement = (best_loss - current_loss) / best_loss
    
    if relative_improvement < threshold:
        wait += 1
        if wait >= patience:
            lr = lr * factor
            wait = 0
    else:
        best_loss = current_loss
        wait = 0
```

优点：
- 自动适应不同任务
- 避免手动调整
- 对噪声鲁棒（通过patience）

缺点：
- 需要调整超参数（patience, threshold）
- 可能错过最佳降低时机
- 对初始学习率敏感
</details>

**练习5.7** 分析ZeRO优化器的通信成本。给定：
- 模型参数量 $P$
- GPU数量 $N$
- 带宽 $B$

计算ZeRO-1、ZeRO-2、ZeRO-3的通信量和内存节省。

*提示：考虑前向、反向、参数更新的通信*

<details>
<summary>答案</summary>

**内存占用**（每个GPU）：
- 标准数据并行：$16P$ 字节（FP16参数2P + FP32参数4P + 动量4P + 方差4P + 梯度2P）
- ZeRO-1：$4P + 12P/N$ 字节（优化器状态分片）
- ZeRO-2：$2P + 14P/N$ 字节（+梯度分片）
- ZeRO-3：$16P/N$ 字节（全部分片）

**通信量**（每步）：
- ZeRO-1：$2P$ （all-reduce梯度）
- ZeRO-2：$2P$ （scatter-reduce + all-gather）
- ZeRO-3：$3P$ （额外的参数all-gather）

**通信时间**：
- ZeRO-1, ZeRO-2：$\frac{2P}{B \cdot N}$
- ZeRO-3：$\frac{3P}{B \cdot N}$

权衡：ZeRO-3内存最省但通信最多，适合带宽充足的场景
</details>

**练习5.8** 设计一个结合一阶和二阶信息的混合优化器。要求：
- 在优化初期使用一阶方法（快速）
- 接近收敛时切换到二阶方法（精确）
- 自动判断切换时机

*提示：可以监控梯度范数或损失变化率*

<details>
<summary>答案</summary>

**混合优化器设计**：

1. **切换条件**：
   - 梯度范数：$\|\nabla f\| < \epsilon_g$
   - 损失变化率：$|f_t - f_{t-1}|/|f_{t-1}| < \epsilon_f$
   - 迭代次数：$t > t_{min}$

2. **算法**：
```
if t < t_min or ||∇f|| > ε_g:
    # 使用Adam（一阶）
    θ = Adam_update(θ, ∇f)
else:
    # 使用L-BFGS（二阶）
    if mod(t, update_freq) == 0:
        H_inv = approximate_hessian_inverse()
    θ = θ - η * H_inv @ ∇f
```

3. **实现细节**：
   - 维护历史梯度用于L-BFGS
   - 平滑切换：逐渐增加二阶信息权重
   - 内存限制：只保存最近m个梯度对

优点：结合快速探索和精确收敛
缺点：切换时机难以确定，需要额外内存
</details>

## 常见陷阱与调试技巧

### 1. 学习率选择错误

**问题**：学习率过大导致发散，过小导致收敛缓慢

**调试技巧**：
- 使用学习率范围测试（LR Range Test）
- 从小学习率开始，指数增长，观察损失变化
- 最佳学习率通常在损失开始发散前的位置

### 2. 梯度爆炸/消失

**问题**：深层网络中梯度指数级增长或衰减

**解决方法**：
- 梯度裁剪：`clip_grad_norm_(parameters, max_norm=1.0)`
- 合适的初始化（Xavier/He初始化）
- 使用归一化技术（BN/LN）
- 残差连接

### 3. Adam的泛化问题

**问题**：Adam训练损失低但测试性能差

**解决方法**：
- 使用AdamW（权重衰减解耦）
- 最后阶段切换到SGD
- 调整 $\beta_2$（如0.98而非0.999）
- 使用更强的正则化

### 4. 批归一化的陷阱

**问题**：训练和测试性能差异大

**常见原因**：
- 批量大小太小（<16）
- 忘记设置model.eval()模式
- moving average统计量更新不当
- 数据分布shift

**解决方法**：
- 使用组归一化或层归一化
- 确保足够的批量大小
- 调整momentum参数（默认0.1）

### 5. 大批量训练的困难

**问题**：增大批量后性能下降

**解决方法**：
- 线性缩放学习率
- 更长的预热期（warmup）
- 使用LARS/LAMB优化器
- 调整正则化强度

### 6. 混合精度训练不稳定

**问题**：FP16训练出现NaN或Inf

**解决方法**：
- 使用动态损失缩放
- 切换到BF16（如果硬件支持）
- 检查模型中的数值不稳定操作
- 在关键层保持FP32（如层归一化）

### 调试工具推荐

1. **TensorBoard**：可视化损失、梯度、权重分布
2. **梯度检查**：数值梯度vs解析梯度
3. **学习率记录**：确保调度器正常工作
4. **梯度直方图**：检测梯度消失/爆炸
5. **激活值统计**：监控各层输出范围

### 经验法则总结

- 先用小数据集快速迭代
- 从简单模型开始，逐步增加复杂度
- 保持可重复性（固定随机种子）
- 记录所有超参数配置
- 定期保存检查点
- 监控多个指标，不只是损失

记住：深度学习优化是科学也是艺术，需要系统的方法和经验积累。
