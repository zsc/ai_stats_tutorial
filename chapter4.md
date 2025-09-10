# 第4章：神经网络基础

神经网络是现代深度学习的基石。本章将从优化和统计的视角深入探讨神经网络的核心机制，包括反向传播算法的数学原理、激活函数的设计考量、权重初始化的统计学基础，以及梯度传播中的数值稳定性问题。通过本章学习，你将理解神经网络不仅仅是"黑箱"，而是有着坚实数学基础的优化系统。

## 4.1 反向传播算法

反向传播（Backpropagation）是神经网络训练的核心算法，本质上是链式法则在计算图上的高效实现。

### 4.1.1 前向传播与计算图

考虑一个L层的前馈神经网络：

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)} = f^{(l)}(\mathbf{z}^{(l)})$$

其中：
- $\mathbf{a}^{(l)}$ 是第$l$层的激活值
- $\mathbf{W}^{(l)}$ 是第$l$层的权重矩阵
- $\mathbf{b}^{(l)}$ 是第$l$层的偏置向量
- $f^{(l)}$ 是第$l$层的激活函数
- $\mathbf{a}^{(0)} = \mathbf{x}$ 是输入

```
输入层        隐藏层1       隐藏层2        输出层
  x₁ ───┬──── h₁₁ ────┬──── h₂₁ ────┬──── y₁
        │             │             │
  x₂ ───┼──── h₁₂ ────┼──── h₂₂ ────┼──── y₂
        │             │             │
  x₃ ───┴──── h₁₃ ────┴──── h₂₃ ────┘
```

### 4.1.2 反向传播的数学推导

给定损失函数 $\mathcal{L}$，反向传播计算每个参数的梯度。定义第$l$层的误差信号：

$$\boldsymbol{\delta}^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$$

**关键递推关系**：

1. **输出层误差**：
   $$\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}^{(L)}}\mathcal{L} \odot f'^{(L)}(\mathbf{z}^{(L)})$$

2. **误差反向传播**：
   $$\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \odot f'^{(l)}(\mathbf{z}^{(l)})$$

3. **参数梯度**：
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

### 4.1.3 计算复杂度分析

设第$l$层有$n_l$个神经元：
- **前向传播**：$O(\sum_{l=1}^L n_l n_{l-1})$
- **反向传播**：$O(\sum_{l=1}^L n_l n_{l-1})$
- **内存需求**：$O(\sum_{l=1}^L n_l)$（需要存储所有激活值）

**Rule of thumb**: 反向传播的计算成本约为前向传播的2-3倍（考虑到梯度计算和参数更新）。

### 4.1.4 自动微分与现代实现

现代深度学习框架使用自动微分（Automatic Differentiation, AD）实现反向传播：

1. **静态图** vs **动态图**：
   - 静态图（如TensorFlow 1.x）：先构建计算图，后执行
   - 动态图（如PyTorch）：即时构建和执行，更灵活

2. **梯度检查点**（Gradient Checkpointing）：
   - 用时间换空间：重新计算前向传播而非存储所有激活值
   - 内存减少：$O(\sqrt{L})$ 而非 $O(L)$
   - 计算增加：约1.33倍前向传播时间

## 4.2 激活函数的选择

激活函数引入非线性，是神经网络表达能力的关键。选择合适的激活函数需要考虑梯度传播、计算效率和统计特性。

### 4.2.1 经典激活函数

1. **Sigmoid函数**：
   $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
   - 优点：输出范围$(0,1)$，可解释为概率
   - 缺点：梯度消失（$\sigma'(x) \leq 0.25$），非零中心

2. **Tanh函数**：
   $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
   - 优点：零中心，梯度较大（$\tanh'(x) \leq 1$）
   - 缺点：仍有梯度消失问题

3. **ReLU函数**：
   $$\text{ReLU}(x) = \max(0, x)$$
   - 优点：计算简单，无梯度消失，稀疏激活
   - 缺点：死神经元问题（Dead ReLU）

### 4.2.2 现代激活函数

1. **Leaky ReLU与参数化ReLU**：
   $$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$
   其中$\alpha$是小的正数（如0.01）或可学习参数（PReLU）

2. **ELU（Exponential Linear Unit）**：
   $$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$
   - 负值时有非零梯度
   - 输出均值接近零

3. **GELU（Gaussian Error Linear Unit）**：
   $$\text{GELU}(x) = x \cdot \Phi(x)$$
   其中$\Phi(x)$是标准正态分布的CDF
   - Transformer的默认选择
   - 平滑且可微

4. **Swish/SiLU**：
   $$\text{Swish}(x) = x \cdot \sigma(\beta x)$$
   - 自门控机制
   - 在深层网络中表现优异

### 4.2.3 激活函数选择准则

**Rule of thumb**：
- **浅层网络**（< 5层）：ReLU通常足够
- **深层网络**（> 10层）：考虑GELU、Swish或ELU
- **循环网络**：Tanh或LSTM/GRU的门控机制
- **输出层**：回归用线性，二分类用Sigmoid，多分类用Softmax

### 4.2.4 激活函数的统计性质

理想的激活函数应保持：
1. **零均值输出**：减少后续层的偏移
2. **单位方差**：稳定梯度传播
3. **Lipschitz连续**：保证优化稳定性

## 4.3 初始化策略

权重初始化直接影响训练动力学和收敛速度。良好的初始化应保持前向传播的信号强度和反向传播的梯度大小。

### 4.3.1 初始化的理论基础

考虑线性层 $y = Wx$，假设：
- $x_i$ 独立同分布，$\mathbb{E}[x_i] = 0$，$\text{Var}(x_i) = 1$
- $W_{ij}$ 独立同分布，$\mathbb{E}[W_{ij}] = 0$

则输出的方差：
$$\text{Var}(y_i) = \sum_j \text{Var}(W_{ij}) \text{Var}(x_j) = n_{\text{in}} \cdot \text{Var}(W_{ij})$$

为保持方差不变，需要：
$$\text{Var}(W_{ij}) = \frac{1}{n_{\text{in}}}$$

### 4.3.2 经典初始化方法

1. **Xavier/Glorot初始化**（适用于Sigmoid/Tanh）：
   $$W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$
   或正态分布：
   $$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

2. **He/Kaiming初始化**（适用于ReLU）：
   $$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$
   考虑了ReLU将一半神经元置零的效应

3. **LSUV（Layer-Sequential Unit-Variance）初始化**：
   - 逐层初始化，确保每层输出方差为1
   - 使用小批量数据进行校准

### 4.3.3 特殊结构的初始化

1. **残差连接**：
   - 残差分支初始化为较小值
   - 或最后一层初始化为零（零初始化残差）

2. **注意力机制**：
   - Query、Key矩阵：$\mathcal{N}(0, 1/d_k)$
   - Value、Output矩阵：标准Xavier初始化

3. **层归一化参数**：
   - $\gamma$ 初始化为1
   - $\beta$ 初始化为0

### 4.3.4 初始化的实用建议

**Rule of thumb**：
- **默认选择**：He初始化（ReLU类）或Xavier初始化（Sigmoid/Tanh）
- **深层网络**：考虑LSUV或渐进式初始化
- **迁移学习**：使用预训练权重，仅初始化新增层
- **调试技巧**：检查初始前向传播的激活值分布，应接近单位方差

## 4.4 梯度消失与爆炸问题

深层网络中，梯度在反向传播过程中可能指数级衰减（消失）或增长（爆炸），严重影响训练。

### 4.4.1 问题的数学分析

考虑L层网络，梯度的传播：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \prod_{l=2}^{L} \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{a}^{(l-1)}}$$

每一项 $\frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{a}^{(l-1)}}$ 包含：
$$\frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{a}^{(l-1)}} = \text{diag}(f'^{(l)}(\mathbf{z}^{(l)})) \mathbf{W}^{(l)}$$

如果 $\|\frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{a}^{(l-1)}}\| < 1$，梯度指数衰减（消失）
如果 $\|\frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{a}^{(l-1)}}\| > 1$，梯度指数增长（爆炸）

### 4.4.2 梯度消失的解决方案

1. **残差连接**（Residual Connections）：
   $$\mathbf{a}^{(l)} = \mathbf{a}^{(l-1)} + \mathcal{F}(\mathbf{a}^{(l-1)})$$
   梯度传播：
   $$\frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{a}^{(l-1)}} = \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{a}^{(l-1)}}$$
   恒等映射保证梯度至少为1

2. **层归一化**（Layer Normalization）：
   $$\text{LN}(\mathbf{x}) = \gamma \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
   - 稳定每层的激活值分布
   - 减少内部协变量偏移

3. **Highway Networks**：
   $$\mathbf{y} = \mathbf{T}(\mathbf{x}) \odot \mathcal{F}(\mathbf{x}) + (1 - \mathbf{T}(\mathbf{x})) \odot \mathbf{x}$$
   其中$\mathbf{T}(\mathbf{x})$是可学习的门控机制

### 4.4.3 梯度爆炸的解决方案

1. **梯度裁剪**（Gradient Clipping）：
   - **按值裁剪**：$g \leftarrow \text{clip}(g, -\theta, \theta)$
   - **按范数裁剪**：$g \leftarrow \frac{g}{\max(1, \|g\|/\theta)} \cdot \theta$
   
   **Rule of thumb**：RNN使用范数裁剪（阈值5-10），CNN/Transformer较少需要

2. **批归一化**（Batch Normalization）：
   $$\text{BN}(\mathbf{x}) = \gamma \frac{\mathbf{x} - \mathbb{E}[\mathbf{x}]}{\sqrt{\text{Var}(\mathbf{x}) + \epsilon}} + \beta$$
   - 训练时使用批统计量
   - 推理时使用移动平均

3. **谱归一化**（Spectral Normalization）：
   $$\mathbf{W}_{\text{SN}} = \frac{\mathbf{W}}{\sigma(\mathbf{W})}$$
   其中$\sigma(\mathbf{W})$是最大奇异值
   - 控制Lipschitz常数
   - 常用于GAN的判别器

### 4.4.4 梯度流的监控与诊断

**实用监控指标**：
1. **梯度范数比率**：$\frac{\|\nabla_{\mathbf{W}^{(l)}}\mathcal{L}\|}{\|\mathbf{W}^{(l)}\|}$
   - 健康范围：$10^{-3}$ 到 $10^{-1}$
   
2. **激活值统计**：
   - 均值应接近0
   - 标准差应接近1
   - 死神经元比例 < 10%

3. **梯度直方图**：
   - 应呈现钟形分布
   - 避免大量零梯度或极值

## 4.5 历史人物：鲁梅尔哈特与反向传播的复兴

大卫·鲁梅尔哈特（David Rumelhart, 1942-2011）是认知科学和神经网络研究的先驱。虽然反向传播算法的数学基础可追溯到60年代，但正是鲁梅尔哈特及其合作者在1986年的开创性工作，使这一算法成为训练神经网络的标准方法。

### 关键贡献

1. **《并行分布式处理》（1986）**：与麦克莱兰德(McClelland)合著的两卷本著作，系统阐述了连接主义理论和反向传播算法。

2. **误差传播的清晰表述**：首次将反向传播表述为一个通用的学习规则，适用于任意前馈网络结构。

3. **认知建模**：展示了神经网络如何解释人类认知现象，如过去时学习中的规则与例外。

### 历史意义

鲁梅尔哈特的工作结束了AI的第一个"寒冬"（1974-1980），重新点燃了对神经网络的研究兴趣。他的贡献不仅在于算法本身，更在于建立了连接主义与符号主义的桥梁，为现代深度学习奠定了理论基础。

**有趣事实**：鲁梅尔哈特最初是心理学家，他将心理学的实验方法引入神经网络研究，强调模型必须能解释人类行为数据。

## 4.6 现代连接：Transformer中的残差连接与层归一化设计

Transformer架构的成功很大程度上归功于其精心设计的残差连接和层归一化策略，这些技术直接解决了深层网络的梯度传播问题。

### 4.6.1 Transformer的残差设计

Transformer采用"Post-LN"或"Pre-LN"两种设计：

**Post-LN（原始Transformer）**：
$$\mathbf{x}_{l+1} = \text{LN}(\mathbf{x}_l + \text{SubLayer}(\mathbf{x}_l))$$

**Pre-LN（现代变体）**：
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{SubLayer}(\text{LN}(\mathbf{x}_l))$$

Pre-LN的优势：
- 更稳定的训练，无需预热学习率
- 梯度直接通过残差路径传播
- 可训练1000+层的模型

### 4.6.2 层归一化的关键作用

Transformer选择层归一化而非批归一化的原因：
1. **序列长度可变**：批统计量不稳定
2. **注意力机制**：需要保持序列内的相对关系
3. **并行化**：层归一化可独立计算每个样本

**RMSNorm优化**：
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \gamma$$
其中 $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$
- 去除均值中心化，计算更高效
- 在大模型中性能相当

### 4.6.3 深度缩放与初始化

**深度缩放**（Depth Scaling）：
对于L层Transformer，残差分支乘以$1/\sqrt{L}$：
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \frac{1}{\sqrt{L}} \text{SubLayer}(\mathbf{x}_l)$$

这保证了：
- 输出方差不随深度增长
- 梯度传播更稳定
- 可训练极深网络（GPT-3达到96层）

### 4.6.4 现代优化技巧

1. **ReZero初始化**：
   $$\mathbf{x}_{l+1} = \mathbf{x}_l + \alpha_l \cdot \text{SubLayer}(\mathbf{x}_l)$$
   其中$\alpha_l$初始化为0，可学习

2. **Fixup初始化**：
   - 特殊的权重初始化
   - 无需归一化层即可训练深层网络

3. **Sandwich LN**：
   在子层前后都加归一化：
   $$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{LN}(\text{SubLayer}(\text{LN}(\mathbf{x}_l)))$$

**Rule of thumb**：
- 小模型（<12层）：Post-LN + 学习率预热
- 大模型（>24层）：Pre-LN 或 Sandwich LN
- 极深模型（>48层）：加入深度缩放

## 本章小结

本章深入探讨了神经网络的基础机制，从优化和统计的角度理解了其工作原理：

### 核心概念

1. **反向传播算法**：
   - 链式法则的高效实现
   - 计算复杂度与前向传播相当
   - 自动微分使实现变得简单

2. **激活函数**：
   - 引入非线性是表达能力的关键
   - ReLU类函数解决了梯度消失
   - GELU/Swish在深层网络中表现更好

3. **初始化策略**：
   - Xavier初始化：适用于Sigmoid/Tanh
   - He初始化：适用于ReLU
   - 保持方差稳定是关键原则

4. **梯度问题与解决**：
   - 残差连接：创建梯度高速公路
   - 归一化技术：稳定激活分布
   - 梯度裁剪：防止数值爆炸

### 关键公式

- 误差反向传播：$\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \odot f'^{(l)}(\mathbf{z}^{(l)})$
- He初始化方差：$\text{Var}(W) = \frac{2}{n_{\text{in}}}$
- 残差连接：$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)$
- 层归一化：$\text{LN}(\mathbf{x}) = \gamma \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$

### 实用准则

- **网络深度**：浅层用ReLU，深层考虑GELU和残差连接
- **初始化选择**：默认He初始化（ReLU）或Xavier初始化（Tanh）
- **梯度监控**：梯度范数比率保持在$10^{-3}$到$10^{-1}$之间
- **归一化策略**：CNN用BN，RNN/Transformer用LN

这些基础为理解现代深度学习架构（如Transformer、BERT、GPT）提供了必要的理论工具。

## 练习题

### 基础题

**练习4.1** 考虑一个简单的两层神经网络：$\mathbf{h} = \text{ReLU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)$，$y = \mathbf{W}_2\mathbf{h} + b_2$。给定损失函数$\mathcal{L} = \frac{1}{2}(y - t)^2$，推导$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1}$的表达式。

*提示*：注意ReLU的导数是分段函数。

<details>
<summary>答案</summary>

首先计算各层的梯度：
1. 输出层误差：$\delta_2 = y - t$
2. 隐藏层误差：$\boldsymbol{\delta}_1 = \mathbf{W}_2^T \delta_2 \odot \mathbf{1}[\mathbf{W}_1\mathbf{x} + \mathbf{b}_1 > 0]$
3. 权重梯度：$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \boldsymbol{\delta}_1 \mathbf{x}^T$

其中$\mathbf{1}[\cdot]$是指示函数，当条件满足时为1，否则为0。
</details>

**练习4.2** 证明对于Xavier初始化，如果输入和权重都是零均值且独立，则线性层的输出也是零均值，且方差为1。

*提示*：使用期望和方差的性质。

<details>
<summary>答案</summary>

设$y_i = \sum_j W_{ij}x_j$，其中$\mathbb{E}[x_j] = 0$，$\text{Var}(x_j) = 1$，$\mathbb{E}[W_{ij}] = 0$。

均值：$\mathbb{E}[y_i] = \sum_j \mathbb{E}[W_{ij}]\mathbb{E}[x_j] = 0$

方差：由于独立性，
$$\text{Var}(y_i) = \sum_j \text{Var}(W_{ij})\text{Var}(x_j) = n_{\text{in}} \cdot \text{Var}(W_{ij})$$

Xavier初始化设置$\text{Var}(W_{ij}) = \frac{1}{n_{\text{in}}}$，因此$\text{Var}(y_i) = 1$。
</details>

**练习4.3** 一个10层的神经网络，每层的梯度缩放因子为0.9。计算从输出层到第一层的梯度衰减比例。这说明了什么问题？

*提示*：考虑指数衰减。

<details>
<summary>答案</summary>

梯度衰减比例 = $(0.9)^{10} \approx 0.349$

这意味着第一层只接收到约35%的梯度信号，说明了梯度消失问题。对于更深的网络（如100层），衰减将是$(0.9)^{100} \approx 2.7 \times 10^{-5}$，几乎没有梯度信号能传到底层，这就是为什么需要残差连接等技术。
</details>

**练习4.4** 比较Sigmoid和ReLU激活函数的梯度。为什么ReLU能缓解梯度消失问题？

*提示*：计算并比较两者的导数范围。

<details>
<summary>答案</summary>

Sigmoid导数：$\sigma'(x) = \sigma(x)(1-\sigma(x))$，最大值为0.25（当$x=0$时）

ReLU导数：
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

ReLU的优势：
1. 正区域梯度恒为1，不会衰减
2. 计算简单，只需判断正负
3. 产生稀疏激活（约50%神经元输出为0）

但ReLU也有死神经元问题，这是为什么有Leaky ReLU等变体。
</details>

### 挑战题

**练习4.5** 设计一个实验来验证He初始化相比于标准正态初始化的优势。描述实验设置、评估指标和预期结果。

*提示*：考虑不同深度的网络和激活值/梯度的分布。

<details>
<summary>答案</summary>

实验设计：
1. **网络架构**：构建不同深度的全连接网络（5、10、20、50层），每层100个神经元，使用ReLU激活
2. **初始化对比**：
   - 标准正态：$\mathcal{N}(0, 1)$
   - He初始化：$\mathcal{N}(0, \sqrt{2/n_{\text{in}}})$
3. **评估指标**：
   - 前向传播：记录每层激活值的均值和标准差
   - 反向传播：记录每层梯度的范数
   - 训练速度：达到特定损失值所需的迭代次数
4. **预期结果**：
   - He初始化：激活值标准差在各层保持稳定（接近1）
   - 标准初始化：深层激活值方差爆炸或消失
   - He初始化的收敛速度快2-5倍

这个实验展示了合适的初始化如何保持信号在深层网络中的传播。
</details>

**练习4.6** 推导带有残差连接的网络中，第L层对第1层的梯度。解释为什么残差连接能够缓解梯度消失。

*提示*：考虑恒等映射的梯度贡献。

<details>
<summary>答案</summary>

对于残差网络：$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}_l(\mathbf{x}_l)$

从第L层到第l层的梯度：
$$\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_{L-1}} \cdots \frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l}$$

每一项：
$$\frac{\partial \mathbf{x}_{k+1}}{\partial \mathbf{x}_k} = \mathbf{I} + \frac{\partial \mathcal{F}_k}{\partial \mathbf{x}_k}$$

展开得：
$$\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \prod_{k=l}^{L-1} \left(\mathbf{I} + \frac{\partial \mathcal{F}_k}{\partial \mathbf{x}_k}\right)$$

关键洞察：即使$\frac{\partial \mathcal{F}_k}{\partial \mathbf{x}_k}$很小，恒等项$\mathbf{I}$保证梯度至少为1，创建了"梯度高速公路"。
</details>

**练习4.7** 分析批归一化（BN）在训练和推理时的行为差异。为什么需要维护移动平均？会带来什么问题？

*提示*：考虑批大小的影响和分布偏移。

<details>
<summary>答案</summary>

训练时BN：
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
其中$\mu_B$、$\sigma_B^2$是当前批的统计量。

推理时BN：
$$\hat{x} = \frac{x - \mu_{MA}}{\sqrt{\sigma_{MA}^2 + \epsilon}}$$
使用训练时的移动平均$\mu_{MA}$、$\sigma_{MA}^2$。

需要移动平均的原因：
1. 推理时可能只有单个样本，无法计算批统计量
2. 保持训练和推理的一致性

潜在问题：
1. **小批量问题**：批大小太小时，批统计量不稳定
2. **分布偏移**：测试数据分布与训练不同时，移动平均不准确
3. **微调困难**：预训练模型的BN统计量需要重新估计

解决方案：
- 使用Layer Norm（不依赖批统计）
- Group Norm（组内归一化）
- 推理时重新估计统计量（Batch Renormalization）
</details>

**练习4.8**（开放题）现代大语言模型使用的层归一化放在注意力层之前（Pre-LN）而不是之后（Post-LN）。设计一个理论分析框架来解释这种选择的优势。

*提示*：考虑梯度路径和训练稳定性。

<details>
<summary>答案</summary>

理论分析框架：

1. **梯度路径分析**：
   - Post-LN：$\mathbf{x}_{l+1} = \text{LN}(f_l(\mathbf{x}_l) + \mathbf{x}_l)$
     梯度：$\frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l}$ 需要通过LN层
   - Pre-LN：$\mathbf{x}_{l+1} = f_l(\text{LN}(\mathbf{x}_l)) + \mathbf{x}_l$
     梯度：$\frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l} = \mathbf{I} + \frac{\partial f_l}{\partial \mathbf{x}_l}$
   
   Pre-LN保留了干净的恒等路径。

2. **方差分析**：
   设$\text{Var}(\mathbf{x}_l) = \sigma^2$
   - Post-LN：需要分析$f_l(\mathbf{x}_l) + \mathbf{x}_l$的方差，然后归一化
   - Pre-LN：$\text{LN}(\mathbf{x}_l)$保证输入方差为1，$f_l$的输出可控

3. **训练动力学**：
   - Post-LN需要学习率预热，否则早期训练不稳定
   - Pre-LN从一开始就稳定，因为每个子层的输入已归一化

4. **极深网络行为**：
   - 100+层网络中，Pre-LN表现显著优于Post-LN
   - Pre-LN可训练1000+层（如GPT-3的96层）

结论：Pre-LN通过保持干净的残差路径和稳定的前向方差，实现了更稳定的深层网络训练。
</details>

## 常见陷阱与错误

### 1. 梯度检查错误

**陷阱**：使用数值梯度检查时，步长选择不当导致误判。

**正确做法**：
- 使用中心差分：$\frac{f(x+h) - f(x-h)}{2h}$
- 步长选择：$h \approx 10^{-5}$（太大误差大，太小有数值误差）
- 相对误差阈值：$\frac{|\nabla_{\text{analytic}} - \nabla_{\text{numeric}}|}{|\nabla_{\text{analytic}}| + |\nabla_{\text{numeric}}|} < 10^{-7}$

### 2. 激活函数使用不当

**陷阱**：在输出层使用ReLU进行回归任务。

**问题**：ReLU只能输出非负值，限制了模型表达能力。

**正确做法**：
- 回归任务：输出层用线性激活
- 二分类：输出层用Sigmoid
- 多分类：输出层用Softmax

### 3. 初始化与激活函数不匹配

**陷阱**：使用ReLU但用Xavier初始化。

**问题**：Xavier假设激活函数线性，不适合ReLU的单边特性。

**正确匹配**：
- ReLU类 → He初始化
- Tanh/Sigmoid → Xavier初始化
- SELU → LeCun初始化

### 4. 批归一化的错误使用

**陷阱**：在RNN的时间步之间使用批归一化。

**问题**：不同时间步的统计量不同，破坏了时序依赖。

**正确做法**：
- RNN使用Layer Norm或不使用归一化
- 批大小很小（<32）时避免BN
- 微调时冻结BN或使用很小的动量

### 5. 梯度裁剪设置错误

**陷阱**：对所有层使用相同的裁剪阈值。

**问题**：不同层的梯度尺度不同，统一裁剪可能过度限制某些层。

**正确做法**：
- 使用全局范数裁剪而非逐层裁剪
- 监控裁剪频率，过于频繁说明阈值太小
- RNN通常需要裁剪，CNN/Transformer较少需要

### 6. 残差连接的实现错误

**陷阱**：残差分支和主分支维度不匹配。

```
错误：x + Conv(x)  # 如果Conv改变了通道数
正确：x + Conv(x)  # 确保Conv保持通道数
      或使用1x1卷积调整：Proj(x) + Conv(x)
```

### 7. 死神经元诊断失败

**陷阱**：没有监控ReLU的死神经元比例。

**诊断方法**：
- 统计激活值为0的比例
- 健康网络：10-50%稀疏度
- 问题信号：>90%死神经元

**解决方案**：
- 降低学习率
- 使用Leaky ReLU
- 调整初始化

### 调试技巧总结

1. **先过拟合小数据**：确保模型有学习能力
2. **梯度监控**：绘制各层梯度直方图
3. **激活值检查**：确保不全为0或爆炸
4. **简化再复杂**：从简单模型逐步增加复杂度
5. **对比基准**：与已知良好的配置对比