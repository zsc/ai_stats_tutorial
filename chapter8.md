# 第8章：Transformer与注意力机制

Transformer架构彻底改变了序列建模的范式，从循环结构转向了纯注意力机制。本章将从数学原理出发，深入理解自注意力的本质，探讨位置编码的设计哲学，分析多头注意力的表达能力，并讨论计算复杂度的优化策略。我们将看到，Transformer不仅是一个技术突破，更代表了对序列信息处理的全新思考方式。

## 8.1 自注意力的数学原理

自注意力机制是Transformer的核心创新，它允许序列中的每个位置直接关注所有其他位置，突破了RNN的顺序处理限制。

### 8.1.1 从序列到序列的映射

考虑输入序列 $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n] \in \mathbb{R}^{d \times n}$，其中每个 $\mathbf{x}_i \in \mathbb{R}^d$ 是位置 $i$ 的特征向量。自注意力的目标是产生输出序列 $\mathbf{Y} = [\mathbf{y}_1, \mathbf{y}_2, ..., \mathbf{y}_n]$，其中每个输出都是所有输入的加权组合：

$$\mathbf{y}_i = \sum_{j=1}^n \alpha_{ij} \mathbf{v}_j$$

这里 $\alpha_{ij}$ 表示位置 $i$ 对位置 $j$ 的注意力权重，$\mathbf{v}_j$ 是位置 $j$ 的值向量。

**关键洞察**：与RNN不同，计算 $\mathbf{y}_i$ 不需要依赖 $\mathbf{y}_{i-1}$，所有位置可以并行计算。

### 8.1.2 注意力分数的计算

注意力权重通过查询(Query)、键(Key)和值(Value)三个线性变换得到：

$$\mathbf{Q} = \mathbf{W}^Q \mathbf{X}, \quad \mathbf{K} = \mathbf{W}^K \mathbf{X}, \quad \mathbf{V} = \mathbf{W}^V \mathbf{X}$$

其中 $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d_k \times d}$ 是可学习参数矩阵。

对于位置 $i$ 和 $j$，原始注意力分数计算为：

$$e_{ij} = \mathbf{q}_i^T \mathbf{k}_j$$

这个点积度量了查询 $\mathbf{q}_i$ 和键 $\mathbf{k}_j$ 的相似度。

**几何解释**：点积越大，表示两个向量在高维空间中方向越一致，因此应该给予更多注意力。

### 8.1.3 缩放点积注意力

直接使用点积存在一个问题：当 $d_k$ 较大时，点积的方差会随维度线性增长。假设 $\mathbf{q}$ 和 $\mathbf{k}$ 的分量独立同分布，均值为0，方差为1，则：

$$\text{Var}(\mathbf{q}^T\mathbf{k}) = d_k$$

大的方差会导致softmax函数进入饱和区，梯度消失。因此引入缩放因子：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**Rule of thumb**: 缩放因子选择 $\sqrt{d_k}$ 使得点积的方差保持为1，避免softmax饱和。

完整的计算流程：
```
输入序列 X
    ↓
线性变换得到 Q, K, V
    ↓
计算缩放点积 QK^T/√d_k
    ↓
Softmax归一化得到注意力权重
    ↓
加权求和 V
    ↓
输出序列 Y
```

### 8.1.4 注意力的概率解释

从概率角度看，注意力机制可以理解为一个软选择过程。给定查询 $\mathbf{q}_i$，我们想从所有键中选择最相关的信息：

$$P(j|i) = \frac{\exp(e_{ij}/\sqrt{d_k})}{\sum_{k=1}^n \exp(e_{ik}/\sqrt{d_k})}$$

这定义了一个在位置上的概率分布。输出 $\mathbf{y}_i$ 是值向量的期望：

$$\mathbf{y}_i = \mathbb{E}_{j \sim P(\cdot|i)}[\mathbf{v}_j] = \sum_{j=1}^n P(j|i) \mathbf{v}_j$$

**信息论视角**：注意力分布的熵 $H(P(\cdot|i))$ 反映了模型的不确定性：
- 低熵：模型聚焦于少数位置（硬注意力）
- 高熵：模型均匀关注所有位置（软注意力）

**统计解释**：自注意力执行了一种非参数密度估计，通过相似度加权来聚合局部信息。

## 8.2 位置编码设计

### 8.2.1 为什么需要位置信息

自注意力机制有一个根本性质：**置换不变性**。对于任意置换矩阵 $\mathbf{P}$：

$$\text{Attention}(\mathbf{Q}\mathbf{P}, \mathbf{K}\mathbf{P}, \mathbf{V}\mathbf{P}) = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})\mathbf{P}$$

这意味着如果我们打乱输入序列的顺序，输出也会以相同方式打乱。虽然这在处理集合时是理想性质，但对序列建模是灾难性的——模型无法区分"狗咬人"和"人咬狗"。

**解决方案**：向输入嵌入中注入位置信息：

$$\mathbf{x}'_i = \mathbf{x}_i + \mathbf{p}_i$$

其中 $\mathbf{p}_i$ 是位置 $i$ 的编码向量。

### 8.2.2 正弦位置编码

Transformer原文提出的正弦位置编码定义为：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

其中 $pos$ 是位置，$i$ 是维度索引。

**设计动机**：
1. **有界性**：正弦函数输出在[-1, 1]范围内，避免数值问题
2. **唯一性**：每个位置获得唯一的编码模式
3. **相对位置**：利用三角恒等式，模型可以学习相对位置关系

对于固定偏移 $k$，位置 $pos+k$ 的编码可以表示为 $pos$ 编码的线性组合：

$$PE_{pos+k} = PE_{pos} \cdot \mathbf{M}_k$$

其中 $\mathbf{M}_k$ 是只依赖于 $k$ 的旋转矩阵。

**频率选择的直觉**：
- 低频成分（大波长）：编码全局位置信息
- 高频成分（小波长）：编码局部相对关系
- 指数级频率间隔：覆盖多个尺度的位置模式

```
位置编码可视化（前8维）：
Pos  Dim0   Dim1   Dim2   Dim3   Dim4   Dim5   Dim6   Dim7
0    0.00   1.00   0.00   1.00   0.00   1.00   0.00   1.00
1    0.84   0.54   0.10   0.99   0.01   1.00   0.00   1.00
2    0.91  -0.42   0.20   0.98   0.02   1.00   0.00   1.00
3    0.14  -0.99   0.30   0.96   0.03   1.00   0.00   1.00
```

### 8.2.3 可学习位置编码

另一种方法是将位置编码作为可学习参数：

$$\mathbf{P} = [\mathbf{p}_1, \mathbf{p}_2, ..., \mathbf{p}_{max}] \in \mathbb{R}^{d \times max}$$

**优点**：
- 灵活性：模型可以学习任务特定的位置模式
- 简单性：实现直接，无需设计编码函数

**缺点**：
- 泛化性：无法处理训练时未见过的长度
- 参数量：需要 $O(max \cdot d)$ 额外参数

**Rule of thumb**: 当序列长度固定或有上界时用可学习编码；需要外推到更长序列时用正弦编码。

### 8.2.4 相对位置编码

相对位置编码直接建模位置间的相对关系，而非绝对位置：

$$e_{ij} = \mathbf{q}_i^T \mathbf{k}_j + \mathbf{q}_i^T \mathbf{r}_{i-j}$$

其中 $\mathbf{r}_{i-j}$ 是相对位置 $i-j$ 的编码。

**T5的简化版本**：使用可学习的标量偏置：

$$e_{ij} = \mathbf{q}_i^T \mathbf{k}_j + b_{i-j}$$

**优势**：
1. **平移不变性**：相同相对位置的交互模式可复用
2. **长度泛化**：自然支持变长序列
3. **归纳偏置**：编码了"近处比远处重要"的先验

**旋转位置编码(RoPE)**：通过旋转操作编码相对位置：

$$\mathbf{q}_i' = \mathbf{R}_i \mathbf{q}_i, \quad \mathbf{k}_j' = \mathbf{R}_j \mathbf{k}_j$$

其中 $\mathbf{R}_i$ 是依赖于位置 $i$ 的旋转矩阵，使得：

$$(\mathbf{R}_i \mathbf{q})^T (\mathbf{R}_j \mathbf{k}) = \mathbf{q}^T \mathbf{R}_{j-i} \mathbf{k}$$

点积只依赖于相对位置 $j-i$。

## 8.3 多头注意力

### 8.3.1 投影与子空间

多头注意力的核心思想是在不同的表示子空间中并行执行注意力：

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}^O$$

其中每个头计算为：

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

参数维度：
- $\mathbf{W}_i^Q, \mathbf{W}_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $\mathbf{W}_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $\mathbf{W}^O \in \mathbb{R}^{hd_v \times d_{model}}$

通常设置 $d_k = d_v = d_{model}/h$，保持总计算量不变。

**几何解释**：每个头将输入投影到低维子空间，在该空间中学习特定的注意力模式，然后将结果组合。

### 8.3.2 多头机制的动机

**1. 表达能力增强**

单头注意力对每个查询产生一个注意力分布。多头允许模型同时关注不同类型的信息：

```
头1: 关注语法依赖（主谓关系）
头2: 关注语义相似（同义词）
头3: 关注位置邻近（局部上下文）
头4: 关注全局模式（段落主题）
```

**2. 学习不同的相似度度量**

不同的投影矩阵学习不同的相似度空间：

$$\text{sim}_i(\mathbf{x}, \mathbf{y}) = (\mathbf{x}\mathbf{W}_i^Q)^T(\mathbf{y}\mathbf{W}_i^K)$$

这相当于在原始空间中使用度量：

$$\text{sim}_i(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T(\mathbf{W}_i^Q)^T\mathbf{W}_i^K\mathbf{y} = \mathbf{x}^T\mathbf{M}_i\mathbf{y}$$

每个头学习不同的度量矩阵 $\mathbf{M}_i$。

**3. 稳定性与冗余**

多头提供了一种集成效应：即使某些头学习到次优模式，其他头可以补偿。实证观察表明，通常只有部分头是关键的，其余提供冗余。

### 8.3.3 参数共享与效率

**参数效率分析**：

单头注意力参数量：$3d_{model}^2$（用于Q、K、V投影）

多头注意力参数量：$3d_{model}^2 + d_{model}^2 = 4d_{model}^2$（额外的输出投影）

尽管参数略有增加，但计算复杂度保持相同：

$$O(n^2 \cdot d_{model})$$

**并行化优势**：

```
传统实现（串行）：
for i in range(h):
    compute head_i
combine all heads

优化实现（并行）：
将QKV投影合并为单个矩阵乘法
reshape为(batch, seq_len, h, d_k)
并行计算所有头的注意力
reshape回(batch, seq_len, d_model)
```

**Rule of thumb**: 
- 小模型：4-8个头
- 基础模型：8-12个头  
- 大模型：16-32个头
- 头数过多会降低每个头的容量（$d_k$ 太小）

### 8.3.4 头之间的相互作用

**注意力模式的多样性**

理想情况下，不同的头应该学习互补的模式。可以通过注意力矩阵的相似度衡量：

$$\text{diversity} = 1 - \frac{1}{h(h-1)}\sum_{i \neq j} \text{cos}(\mathbf{A}_i, \mathbf{A}_j)$$

其中 $\mathbf{A}_i$ 是第 $i$ 个头的注意力矩阵。

**头的专门化现象**

研究发现，训练后的模型中不同头会自发专门化：
- **位置头**：关注固定相对位置（如前一个token）
- **语法头**：捕捉语法结构（如依存关系）
- **稀有词头**：专注于低频词汇
- **全局头**：均匀分布的注意力

**头的重要性分析**

通过剪枝实验可以识别关键头：

$$\text{importance}_i = \|\mathbf{W}^O_{:,i \cdot d_v:(i+1) \cdot d_v}\|_F$$

实践中，通常只有30-50%的头对最终性能至关重要。

**交互效应**

输出投影 $\mathbf{W}^O$ 不仅组合不同头的输出，还学习它们之间的交互：

$$\mathbf{y} = \sum_{i=1}^h \text{head}_i \mathbf{W}^O_i + \text{interactions}$$

这种交互使得模型能够根据不同头的置信度动态加权。

## 8.4 计算复杂度分析

### 8.4.1 时间复杂度

自注意力的时间复杂度由三个主要操作决定：

**1. 计算QKV投影**：
$$O(n \cdot d_{model}^2)$$

**2. 计算注意力分数**：
$$O(n^2 \cdot d_k)$$

**3. 加权求和**：
$$O(n^2 \cdot d_v)$$

总复杂度：$O(n^2 \cdot d_{model} + n \cdot d_{model}^2)$

**与RNN的对比**：
- RNN: $O(n \cdot d_{model}^2)$ - 线性于序列长度
- Transformer: $O(n^2 \cdot d_{model})$ - 二次于序列长度

**复杂度权衡**：
```
序列长度 n < d_model 时：Transformer更高效
序列长度 n > d_model 时：RNN更高效
典型设置：n ≈ 512, d_model ≈ 768，两者相当
```

### 8.4.2 空间复杂度

主要内存消耗来自注意力矩阵：

**前向传播**：
- 注意力分数矩阵：$O(n^2 \cdot h)$
- QKV矩阵：$O(n \cdot d_{model})$
- 总计：$O(n^2 \cdot h + n \cdot d_{model})$

**反向传播**：
需要存储所有中间激活，空间复杂度：
$$O(L \cdot n^2 \cdot h)$$

其中 $L$ 是层数。

**内存优化技术**：
1. **梯度检查点**：只存储部分激活，需要时重计算
2. **混合精度**：使用FP16减少内存占用
3. **注意力分块**：将注意力矩阵分块计算

**Rule of thumb**: 
- 批大小1，序列长度2048，12层模型约需16GB显存
- 序列长度翻倍，内存需求增加4倍

### 8.4.3 并行化优势

Transformer的关键优势是高度并行化：

**序列维度并行**：
```
RNN必须顺序处理：
h_1 → h_2 → h_3 → ... → h_n
时间复杂度: O(n)

Transformer可并行处理：
[h_1, h_2, h_3, ..., h_n] 同时计算
时间复杂度: O(1) (给定足够并行资源)
```

**矩阵运算效率**：
自注意力主要是矩阵乘法，可充分利用：
- GPU的张量核心
- TPU的矩阵乘法单元
- CPU的SIMD指令

**批处理效率**：
```
计算效率 = 实际FLOPS / 理论峰值FLOPS

小批量：20-30% (受内存带宽限制)
大批量：70-80% (计算密集)
```

### 8.4.4 长序列的挑战

标准自注意力的二次复杂度限制了处理长序列的能力：

**问题规模**：
```
序列长度  注意力矩阵大小  16-bit存储
512       262K            0.5MB
2K        4M              8MB
8K        64M             128MB
32K       1024M           2GB
128K      16384M          32GB
```

**高效注意力变体**：

**1. 稀疏注意力**：
只计算部分注意力连接：
$$O(n \cdot \sqrt{n} \cdot d)$$

模式包括：
- 局部注意力：只关注附近 $k$ 个位置
- 跨步注意力：每隔 $k$ 个位置采样
- 随机注意力：随机采样连接

**2. 线性注意力**：
通过核方法近似：
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \approx \phi(\mathbf{Q})(\phi(\mathbf{K})^T\mathbf{V})$$

复杂度降至 $O(n \cdot d^2)$

**3. 分层注意力**：
```
原始序列: [x_1, x_2, ..., x_n]
     ↓ 局部注意力
压缩表示: [z_1, z_2, ..., z_{n/k}]
     ↓ 全局注意力
最终输出: [y_1, y_2, ..., y_n]
```

**Flash Attention优化**：
通过分块和重计算减少内存访问：

1. 将QKV分成小块
2. 在SRAM中计算每块的注意力
3. 累积部分结果
4. 避免存储完整的 $n \times n$ 矩阵

效果：
- 内存使用：$O(n)$ 而非 $O(n^2)$
- 实际速度：提升2-4倍
- 支持序列长度：可达64K+

## 8.5 历史人物：瓦斯瓦尼与"Attention is All You Need"

2017年，Google Brain团队的阿希什·瓦斯瓦尼(Ashish Vaswani)等八位研究者发表了改变深度学习历史进程的论文《Attention is All You Need》。这篇仅12页的论文提出了Transformer架构，彻底革新了序列建模的范式。

**背景与动机**

在Transformer之前，序列建模被RNN和LSTM统治。瓦斯瓦尼团队观察到几个关键问题：
1. RNN的顺序依赖阻碍了并行训练
2. 长距离依赖关系难以建模
3. 计算效率在长序列上严重下降

团队的核心洞察是：**注意力机制本身就足够强大，不需要循环或卷积结构**。

**关键创新**

1. **纯注意力架构**：完全抛弃了循环和卷积，只使用注意力和前馈网络
2. **多头机制**：允许模型同时关注不同的信息子空间
3. **位置编码**：巧妙解决了注意力的置换不变性问题
4. **编码器-解码器结构**：为序列到序列任务提供了通用框架

**影响与传承**

Transformer的影响远超最初的机器翻译任务：
- **NLP革命**：BERT、GPT系列、T5等模型都基于Transformer
- **跨领域应用**：Vision Transformer将其推广到计算机视觉
- **规模化定律**：Transformer展现了前所未有的规模化能力
- **产业变革**：ChatGPT、Claude等大语言模型的基础架构

**哲学思考**

瓦斯瓦尼团队的工作体现了科研的几个重要原则：
1. **简化而非复杂化**：去除循环结构看似激进，实则让模型更简洁
2. **归纳偏置的权衡**：放弃RNN的顺序偏置，换取更大的模型容量
3. **工程与理论结合**：既有理论创新，又考虑实际计算效率

"Attention is All You Need"不仅是技术突破，更是思维范式的转变——从局部处理到全局交互，从顺序计算到并行处理。

## 8.6 现代连接：Flash Attention与长上下文优化

随着大语言模型的发展，处理长上下文成为关键挑战。Flash Attention及其后续发展代表了算法与硬件协同设计的新范式。

**Flash Attention核心思想**

传统注意力计算需要存储完整的 $n \times n$ 注意力矩阵，Flash Attention通过分块计算避免了这一需求：

```
传统方法：
S = QK^T → P = softmax(S) → O = PV
需要存储大矩阵S和P

Flash Attention：
将Q,K,V分成小块
for each block:
    计算局部注意力
    累积到输出
只需存储块大小的矩阵
```

**硬件感知优化**

Flash Attention的设计充分考虑了GPU内存层次：
- **SRAM** (快，小)：~20MB，带宽19TB/s
- **HBM** (慢，大)：~40GB，带宽1.5TB/s

通过最小化HBM访问，实现2-4倍加速。

**Flash Attention 2的改进**

1. **更好的并行策略**：在序列维度和批维度上并行
2. **减少非矩阵运算**：优化softmax等操作
3. **支持各种注意力变体**：因果注意力、分组查询注意力等

**长上下文的其他优化**

**1. RoPE扩展**
```
原始RoPE: 支持2K上下文
位置插值: 线性插值到32K
NTK-aware: 调整基频支持更长序列
YaRN: 结合插值和外推
```

**2. 滑动窗口注意力**
- Mistral的方案：每层使用固定窗口(如4K)
- 通过多层叠加实现长程依赖
- 内存使用线性而非二次增长

**3. 环形注意力(Ring Attention)**
```
将序列分配到多个设备
设备1: [0:n/4]
设备2: [n/4:n/2]
设备3: [n/2:3n/4]
设备4: [3n/4:n]
环形通信计算跨设备注意力
```

**4. 流式注意力**
- 增量处理新token
- 缓存键值对(KV Cache)
- 支持实时生成

**实际应用中的权衡**

**Rule of thumb**：
- 8K以下：标准注意力 + Flash Attention
- 8K-32K：滑动窗口或稀疏注意力
- 32K-128K：环形注意力或混合方案
- 128K+：需要专门的长文本架构

**未来趋势**

1. **亚二次复杂度**：Mamba等架构探索线性复杂度
2. **检索增强**：结合外部记忆减少上下文压力
3. **动态稀疏**：根据内容自适应选择注意力模式
4. **专用硬件**：为Transformer设计的AI芯片

Flash Attention的成功启示：**算法创新必须与硬件特性结合**，这种协同设计思想正在重塑AI系统的优化方法论。

## 8.7 本章小结

本章深入探讨了Transformer架构的核心机制——自注意力，从数学原理到工程实践，揭示了这一革命性架构的精髓。

**核心概念回顾**

1. **自注意力机制**
   - 通过Query-Key-Value框架实现序列内的全局交互
   - 缩放因子 $1/\sqrt{d_k}$ 防止softmax饱和
   - 注意力权重提供了可解释的信息流动模式

2. **位置编码**
   - 解决注意力的置换不变性问题
   - 正弦编码：支持任意长度，包含多尺度信息
   - 相对位置编码：更好的长度泛化能力

3. **多头注意力**
   - 在不同子空间并行学习多种注意力模式
   - 提供集成效应和表达能力增强
   - 典型配置：$h \in [4, 32]$，$d_k = d_{model}/h$

4. **计算复杂度**
   - 时间：$O(n^2 \cdot d + n \cdot d^2)$
   - 空间：$O(n^2 \cdot h \cdot L)$
   - 高度并行化，适合现代硬件

**关键公式汇总**

缩放点积注意力：
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

多头注意力：
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}^O$$

正弦位置编码：
$$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d})$$

**实用准则**

- **模型规模选择**：参数量翻倍，头数增加1.4倍
- **序列长度处理**：<8K用标准注意力，>8K考虑稀疏变体
- **内存估算**：序列长度翻倍，内存需求4倍增长
- **训练稳定性**：使用Pre-LN，warmup学习率，梯度裁剪

**深刻洞察**

1. **从局部到全局**：Transformer突破了RNN/CNN的局部处理限制
2. **归纳偏置的权衡**：用更大的模型容量换取更少的结构假设
3. **硬件友好设计**：矩阵运算为主，充分利用并行计算
4. **可扩展性**：展现了前所未有的规模化能力

Transformer不仅是技术创新，更代表了深度学习的新范式——**通过注意力机制实现灵活、可学习的信息路由**。

## 8.8 练习题

### 基础题

**练习8.1** 解释为什么自注意力需要缩放因子 $1/\sqrt{d_k}$。如果不使用这个缩放因子会发生什么？

*提示：考虑点积的方差如何随维度变化，以及这对softmax函数的影响。*

<details>
<summary>答案</summary>

不使用缩放因子时，点积 $\mathbf{q}^T\mathbf{k}$ 的方差为 $d_k$。当 $d_k$ 较大（如512）时，点积的值可能达到 ±20 或更大。经过softmax后，最大值对应的概率接近1，其他接近0，导致：
1. 梯度消失：softmax导数在饱和区接近0
2. 注意力退化为硬选择，失去软注意力的优势
3. 训练不稳定，学习缓慢

缩放因子使点积方差保持为1，softmax输入在合理范围内，保持梯度流动。
</details>

**练习8.2** 给定序列长度 $n=1024$，模型维度 $d_{model}=768$，头数 $h=12$。计算：
a) 单个注意力层的参数量
b) 处理一个批次（batch_size=32）需要的注意力矩阵内存（FP16）

*提示：考虑QKV投影和输出投影的参数，以及所有头的注意力矩阵。*

<details>
<summary>答案</summary>

a) 参数量：
- QKV投影：$3 \times d_{model} \times d_{model} = 3 \times 768 \times 768 = 1,769,472$
- 输出投影：$d_{model} \times d_{model} = 768 \times 768 = 589,824$
- 总计：$2,359,296$ 参数

b) 注意力矩阵内存：
- 每个头的注意力矩阵：$n \times n = 1024 \times 1024$
- 所有头：$h \times n \times n = 12 \times 1024 \times 1024$
- 批大小32：$32 \times 12 \times 1024 \times 1024$
- FP16存储：$32 \times 12 \times 1024 \times 1024 \times 2$ 字节 = 768MB
</details>

**练习8.3** 证明自注意力机制具有置换不变性：如果输入序列的顺序改变，输出也会以相同方式改变。

*提示：使用置换矩阵 $\mathbf{P}$，证明 $\text{Attention}(\mathbf{XP}) = \text{Attention}(\mathbf{X})\mathbf{P}$。*

<details>
<summary>答案</summary>

设 $\mathbf{P}$ 是置换矩阵，$\mathbf{X}' = \mathbf{XP}$ 是置换后的输入。

计算QKV：
- $\mathbf{Q}' = \mathbf{W}^Q\mathbf{X}' = \mathbf{W}^Q\mathbf{XP} = \mathbf{QP}$
- 同理：$\mathbf{K}' = \mathbf{KP}$，$\mathbf{V}' = \mathbf{VP}$

计算注意力：
$$\text{Attention}(\mathbf{Q}', \mathbf{K}', \mathbf{V}') = \text{softmax}\left(\frac{\mathbf{Q}'\mathbf{K}'^T}{\sqrt{d_k}}\right)\mathbf{V}'$$

$$= \text{softmax}\left(\frac{\mathbf{QP}(\mathbf{KP})^T}{\sqrt{d_k}}\right)\mathbf{VP}$$

$$= \text{softmax}\left(\frac{\mathbf{QPP}^T\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{VP}$$

由于 $\mathbf{PP}^T = \mathbf{I}$：
$$= \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right)\mathbf{VP} = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})\mathbf{P}$$

因此自注意力具有置换不变性。
</details>

### 挑战题

**练习8.4** 设计一个注意力模式，使得每个位置只关注距离不超过 $k$ 的位置（局部注意力）。如何修改标准自注意力来实现这一点？分析其时间复杂度。

*提示：考虑使用掩码矩阵，或重新组织计算顺序。*

<details>
<summary>答案</summary>

方法1：掩码实现
创建掩码矩阵 $\mathbf{M}$，其中 $M_{ij} = -\infty$ 如果 $|i-j| > k$，否则为0。
$$\text{LocalAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

方法2：分块计算
将序列分成重叠的块，每块大小 $2k+1$：
```
for i in range(n):
    start = max(0, i-k)
    end = min(n, i+k+1)
    q_i = Q[i]
    K_local = K[start:end]
    V_local = V[start:end]
    output[i] = attention(q_i, K_local, V_local)
```

时间复杂度：$O(n \cdot k \cdot d)$，当 $k \ll n$ 时显著优于 $O(n^2 \cdot d)$。

内存复杂度：$O(n \cdot k)$ 而非 $O(n^2)$。
</details>

**练习8.5** 推导相对位置编码如何保持平移不变性。具体说明RoPE（旋转位置编码）如何通过旋转操作实现这一性质。

*提示：考虑二维旋转矩阵，以及如何将其推广到高维。*

<details>
<summary>答案</summary>

RoPE的核心思想是使用旋转矩阵编码位置。对于2D情况：

$$\mathbf{R}_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

位置 $m$ 的旋转角度：$\theta_m = m \cdot \omega$，其中 $\omega$ 是基频。

对查询和键应用旋转：
- $\mathbf{q}_m' = \mathbf{R}_{m\omega} \mathbf{q}_m$
- $\mathbf{k}_n' = \mathbf{R}_{n\omega} \mathbf{k}_n$

点积计算：
$$\mathbf{q}_m'^T \mathbf{k}_n' = (\mathbf{R}_{m\omega} \mathbf{q}_m)^T (\mathbf{R}_{n\omega} \mathbf{k}_n)$$
$$= \mathbf{q}_m^T \mathbf{R}_{m\omega}^T \mathbf{R}_{n\omega} \mathbf{k}_n$$
$$= \mathbf{q}_m^T \mathbf{R}_{(n-m)\omega} \mathbf{k}_n$$

关键性质：点积只依赖于相对位置 $(n-m)$，不依赖绝对位置。

高维推广：将 $d$ 维空间分成 $d/2$ 个2D子空间，每个子空间使用不同频率：
$$\omega_i = 10000^{-2i/d}$$

这确保了不同尺度的位置信息都被编码。
</details>

**练习8.6** 分析Flash Attention如何通过分块计算减少内存访问。给定块大小 $B$，推导其IO复杂度。

*提示：考虑HBM和SRAM之间的数据传输量。*

<details>
<summary>答案</summary>

传统注意力的IO复杂度：
1. 读取Q, K, V：$O(Nd)$ 
2. 写入/读取 $\mathbf{S} = \mathbf{QK}^T$：$O(N^2)$
3. 写入/读取 $\mathbf{P} = \text{softmax}(\mathbf{S})$：$O(N^2)$
4. 写入输出：$O(Nd)$
总计：$O(N^2 + Nd)$

Flash Attention分块计算：
- 将Q, K, V分成大小为 $B$ 的块
- 块数：$T = N/B$

对每个输出块 $i$：
1. 加载 $\mathbf{Q}_i$：$O(Bd)$
2. 对每个KV块 $j$：
   - 加载 $\mathbf{K}_j, \mathbf{V}_j$：$O(Bd)$
   - 在SRAM中计算局部注意力
   - 累积到输出
3. 写回输出块：$O(Bd)$

总IO：$O(T^2 \cdot Bd) = O(N^2d/B)$

当 $B = \Theta(\sqrt{M/d})$（$M$是SRAM大小）时，IO复杂度为：
$$O(N^2d^2/M)$$

相比传统方法，当 $d \ll \sqrt{M}$ 时有显著改进。
</details>

**练习8.7**（开放题）设计一种新的位置编码方案，要求：(1)支持变长序列，(2)计算高效，(3)能够表达相对位置关系。说明你的设计理由。

*提示：可以结合现有方法的优点，或从信号处理、图论等领域借鉴思想。*

<details>
<summary>参考思路</summary>

一种可能的设计：**分层频率编码**

核心思想：使用多尺度的频率成分，类似小波变换：

1. **基础层**：低频正弦编码捕捉全局位置
   $$PE_{base}(pos, i) = \sin(pos \cdot 2^{-i})$$

2. **细节层**：高频成分捕捉局部关系
   $$PE_{detail}(pos, i) = \sin(pos \cdot 2^{i-d/2})$$

3. **自适应混合**：根据序列长度动态调整权重
   $$PE(pos) = \alpha(L) \cdot PE_{base} + (1-\alpha(L)) \cdot PE_{detail}$$
   其中 $\alpha(L) = \sigmoid(\log L / \log L_{max})$

优势：
- 短序列：更多高频成分，精确的局部关系
- 长序列：更多低频成分，稳定的全局结构
- 计算高效：仍是O(1)的查表操作
- 相对位置：通过频率差异自然编码

另一种思路：**图拉普拉斯编码**
- 将序列视为链式图
- 使用图拉普拉斯算子的特征向量作为位置编码
- 自然满足平移不变性和局部平滑性
</details>

**练习8.8** 证明多头注意力的表达能力严格强于单头注意力。构造一个单头注意力无法表示但多头可以表示的函数。

*提示：考虑需要同时关注多个不相交位置集合的情况。*

<details>
<summary>答案</summary>

构造示例：考虑需要同时执行"复制第1个token"和"复制最后一个token"的任务。

输入：$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n]$
目标输出：$\mathbf{y}_i = \mathbf{x}_1 + \mathbf{x}_n$ 对所有 $i$

单头注意力的局限：
- 注意力权重必须满足：$\sum_j \alpha_{ij} = 1$
- 无法同时让 $\alpha_{i1} = 0.5$ 和 $\alpha_{in} = 0.5$，其他为0
- 因为这需要注意力矩阵的秩至少为2

双头注意力的解决方案：
- 头1：$\alpha^{(1)}_{ij} = 1$ 如果 $j=1$，否则为0
- 头2：$\alpha^{(2)}_{ij} = 1$ 如果 $j=n$，否则为0
- 输出投影：$\mathbf{W}^O = 0.5 \mathbf{I}$

这样：$\mathbf{y}_i = 0.5(\mathbf{x}_1 + \mathbf{x}_n)$

一般性结论：$h$ 头注意力可以表示秩最多为 $h$ 的注意力模式组合，而单头限制为秩1。
</details>

## 8.9 常见陷阱与错误

### 1. 位置编码的常见错误

**错误**：忘记添加位置编码
```
# 错误：直接使用词嵌入
x = embedding(tokens)
output = transformer(x)

# 正确：添加位置编码
x = embedding(tokens) + positional_encoding
output = transformer(x)
```

**后果**：模型无法区分不同位置的相同词，严重影响性能。

**错误**：位置编码与词嵌入维度不匹配
```
# 错误：维度不一致
embedding_dim = 512
pos_encoding_dim = 256  # 不匹配！

# 正确：保持维度一致
embedding_dim = 512
pos_encoding_dim = 512
```

### 2. 注意力计算的数值问题

**错误**：忘记缩放因子
```
# 错误：直接计算点积
attention_scores = torch.matmul(Q, K.transpose(-2, -1))
attention_weights = softmax(attention_scores)

# 正确：使用缩放因子
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attention_weights = softmax(attention_scores)
```

**后果**：梯度消失，训练极其缓慢或失败。

**错误**：在错误的维度上应用softmax
```
# 错误：在序列维度softmax
attention_weights = softmax(scores, dim=-2)  # 错误！

# 正确：在最后一个维度
attention_weights = softmax(scores, dim=-1)
```

### 3. 掩码处理错误

**错误**：使用0作为padding掩码值
```
# 错误：使用0会影响softmax分布
mask = torch.zeros(seq_len, seq_len)
scores = scores * mask

# 正确：使用负无穷
mask = mask.float().masked_fill(mask == 0, -1e9)
scores = scores + mask
```

**错误**：因果掩码的形状错误
```
# 错误：上三角应该被掩码
causal_mask = torch.triu(torch.ones(n, n))  # 错误方向！

# 正确：下三角保留
causal_mask = torch.tril(torch.ones(n, n))
```

### 4. 多头注意力的实现陷阱

**错误**：头维度计算错误
```
# 错误：不能整除
d_model = 512
num_heads = 7  # 512 % 7 != 0

# 正确：确保整除
d_model = 512
num_heads = 8  # 512 % 8 = 0，d_k = 64
```

**错误**：reshape操作的顺序错误
```
# 错误：维度顺序混乱
x = x.reshape(batch, seq_len, d_model, num_heads)

# 正确：保持正确的维度顺序
x = x.reshape(batch, seq_len, num_heads, d_k)
x = x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
```

### 5. 内存和效率问题

**错误**：存储完整的注意力矩阵
```
# 错误：对长序列会OOM
attention_matrix = torch.zeros(batch_size, num_heads, seq_len, seq_len)
# 对seq_len=10000，需要~3GB内存！

# 正确：使用checkpointing或Flash Attention
with torch.cuda.amp.autocast():
    output = flash_attention(q, k, v)
```

**错误**：不必要的矩阵复制
```
# 错误：创建多个副本
Q = self.q_proj(x).clone()  # 不必要的clone
K = self.k_proj(x).clone()

# 正确：直接使用
Q = self.q_proj(x)
K = self.k_proj(x)
```

### 6. 训练稳定性问题

**错误**：没有使用层归一化
```
# 错误：直接连接
x = x + self_attention(x)

# 正确：使用Pre-LN或Post-LN
# Pre-LN (更稳定)
x = x + self_attention(layer_norm(x))

# Post-LN (原始Transformer)
x = layer_norm(x + self_attention(x))
```

**错误**：学习率过大
```
# 错误：对Transformer使用过大的学习率
optimizer = Adam(lr=1e-2)  # 太大！

# 正确：使用较小的学习率和warmup
optimizer = Adam(lr=5e-4)
scheduler = WarmupScheduler(warmup_steps=4000)
```

### 7. 推理优化的错误

**错误**：重复计算KV缓存
```
# 错误：每个token都重新计算所有KV
for i in range(seq_len):
    k = compute_keys(all_tokens[:i+1])  # 重复计算！
    v = compute_values(all_tokens[:i+1])

# 正确：增量更新KV缓存
if kv_cache is not None:
    k = torch.cat([kv_cache['k'], new_k], dim=1)
    v = torch.cat([kv_cache['v'], new_v], dim=1)
```

### 8. 调试技巧

**检查注意力权重分布**：
```python
# 监控注意力权重的熵
entropy = -(weights * weights.log()).sum(-1).mean()
print(f"Attention entropy: {entropy:.3f}")
# 过低：过度聚焦；过高：注意力分散
```

**验证位置编码效果**：
```python
# 检查相同词在不同位置的表示差异
token_id = 42
pos1_repr = model.encode([token_id], position=0)
pos2_repr = model.encode([token_id], position=10)
similarity = cosine_similarity(pos1_repr, pos2_repr)
# 应该 < 1.0，表明位置信息被编码
```

**监控梯度流**：
```python
# 检查各层梯度范数
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: {grad_norm:.6f}")
# 梯度消失：< 1e-6；梯度爆炸：> 1000
```

### 预防措施清单

1. ✅ 始终添加位置编码
2. ✅ 使用缩放因子 $1/\sqrt{d_k}$
3. ✅ 正确处理padding和因果掩码
4. ✅ 确保维度可被头数整除
5. ✅ 使用混合精度训练节省内存
6. ✅ 实施梯度裁剪防止爆炸
7. ✅ 使用学习率warmup
8. ✅ 监控注意力权重分布
9. ✅ 为长序列使用高效注意力变体
10. ✅ 推理时使用KV缓存