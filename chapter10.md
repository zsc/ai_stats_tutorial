# 第10章：变分自编码器与生成建模

变分自编码器（VAE）优雅地结合了贝叶斯推断与深度学习，为生成模型开辟了新的方向。本章将从统计推断的视角深入理解VAE的原理，掌握其训练技巧，并探讨在现代AI系统中的应用。我们将看到，VAE不仅是一个生成模型，更是理解表示学习的重要框架。

## 10.1 从自编码器到变分自编码器

### 10.1.1 传统自编码器的局限

传统自编码器通过编码器-解码器架构学习数据的压缩表示：

```
输入 x → 编码器 f(x) → 隐变量 z → 解码器 g(z) → 重构 x̂
```

损失函数仅关注重构误差：
$$\mathcal{L}_{AE} = \|x - \hat{x}\|^2$$

**主要问题**：
- 隐空间不连续：相似输入可能映射到相距很远的隐表示
- 无法生成新样本：隐空间中的任意点可能对应无意义的重构
- 缺乏概率解释：无法量化不确定性

### 10.1.2 生成模型的概率视角

VAE从生成过程的角度重新定义问题：

1. **生成过程**：
   - 从先验分布采样隐变量：$z \sim p(z) = \mathcal{N}(0, I)$
   - 通过解码器生成数据：$x \sim p_\theta(x|z)$

2. **推断问题**：
   给定观察数据$x$，推断其对应的隐变量$z$的后验分布$p(z|x)$

关键挑战：真实后验$p(z|x) = \frac{p(x|z)p(z)}{p(x)}$难以计算，因为边际似然$p(x) = \int p(x|z)p(z)dz$通常不可解。

## 10.2 变分推断与ELBO

### 10.2.1 变分推断的核心思想

既然真实后验$p(z|x)$难以计算，我们用一个可处理的分布$q_\phi(z|x)$来近似它：

$$q_\phi(z|x) \approx p(z|x)$$

其中$\phi$是变分参数（编码器的参数）。目标是最小化两个分布之间的KL散度：

$$\text{KL}[q_\phi(z|x) \| p(z|x)] = \mathbb{E}_{q_\phi(z|x)}\left[\log\frac{q_\phi(z|x)}{p(z|x)}\right]$$

### 10.2.2 ELBO的推导

直接优化上述KL散度是困难的，因为它包含未知的$p(z|x)$。通过巧妙的代数变换：

$$\log p(x) = \mathbb{E}_{q_\phi(z|x)}[\log p(x)]$$

$$= \mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p(x,z)}{p(z|x)}\right]$$

$$= \mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p(x,z)}{q_\phi(z|x)} \cdot \frac{q_\phi(z|x)}{p(z|x)}\right]$$

$$= \underbrace{\mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p(x,z)}{q_\phi(z|x)}\right]}_{\text{ELBO}} + \underbrace{\text{KL}[q_\phi(z|x) \| p(z|x)]}_{\geq 0}$$

因此：
$$\log p(x) \geq \text{ELBO} = \mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p(x,z)}{q_\phi(z|x)}\right]$$

### 10.2.3 ELBO的可解释形式

将ELBO展开为更直观的形式：

$$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}[q_\phi(z|x) \| p(z)]$$

两项具有清晰的含义：
- **重构项**：$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ - 确保隐变量能够重构原始数据
- **正则项**：$-\text{KL}[q_\phi(z|x) \| p(z)]$ - 使近似后验接近先验，确保隐空间的规整性

**Rule of Thumb**：重构项与正则项的平衡是VAE训练的关键。初期可让重构项权重较大，后期逐渐增加正则项权重。

## 10.3 VAE的架构与实现细节

### 10.3.1 编码器设计

编码器$q_\phi(z|x)$输出高斯分布的参数：

```
        输入 x
          ↓
      神经网络 f_φ
          ↓
    ┌─────┴─────┐
    ↓           ↓
  μ(x)        σ²(x)
    
q_φ(z|x) = N(μ(x), diag(σ²(x)))
```

**关键设计选择**：
1. 使用对角协方差矩阵降低计算复杂度
2. 输出log方差而非方差本身，保证数值稳定性：
   $$\sigma^2 = \exp(\text{logvar})$$

### 10.3.2 解码器设计

解码器$p_\theta(x|z)$的选择取决于数据类型：

- **连续数据**（如图像强度）：高斯分布
  $$p_\theta(x|z) = \mathcal{N}(\mu_\theta(z), \sigma^2I)$$
  
- **二值数据**（如二值图像）：伯努利分布
  $$p_\theta(x|z) = \text{Bernoulli}(p_\theta(z))$$

### 10.3.3 损失函数的实际形式

对于高斯解码器和高斯编码器：

$$\mathcal{L}_{VAE} = \frac{1}{2}\sum_{i=1}^D \left[(x_i - \hat{x}_i)^2 + \mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1\right]$$

其中$D$是数据维度。

## 10.4 重参数化技巧

### 10.4.1 梯度传播的挑战

从$q_\phi(z|x)$采样是随机操作，无法直接反向传播梯度：

```
不可微的采样过程：
x → 编码器 → (μ, σ²) → 采样 z ~ N(μ, σ²) → 解码器 → x̂
                            ↑
                      梯度无法通过
```

### 10.4.2 重参数化的核心思想

将随机性从参数中分离出来：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中$\odot$表示逐元素乘积。

```
可微的重参数化：
x → 编码器 → (μ, σ²) ──────→ z = μ + σ⊙ε → 解码器 → x̂
                    ↑                ↑
                梯度可传播        ε ~ N(0,I)
```

### 10.4.3 重参数化的一般形式

对于其他分布族，重参数化技巧可以推广：

- **Gumbel-Softmax**：用于离散分布的连续松弛
- **Gamma分布**：使用形状-尺度参数化
- **Beta分布**：通过Kumaraswamy分布近似

**Rule of Thumb**：当需要从复杂分布采样时，优先考虑是否存在可重参数化的形式。

## 10.5 β-VAE与解耦表示

### 10.5.1 解耦表示的动机

理想的表示应该将数据的独立生成因子分离到不同的隐变量维度：

```
原始图像          解耦的隐变量
┌────────┐      z₁: 物体类型
│ 红色   │      z₂: 颜色
│ 圆形   │  →   z₃: 形状
│ 物体   │      z₄: 位置
└────────┘      z₅: 大小
```

### 10.5.2 β-VAE的损失函数

β-VAE通过调整KL项的权重来促进解耦：

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot \text{KL}[q_\phi(z|x) \| p(z)]$$

- $\beta > 1$：增强正则化，促进解耦但可能降低重构质量
- $\beta < 1$：减弱正则化，提高重构质量但降低解耦程度

### 10.5.3 解耦度的定量评估

**互信息间隙（MIG）**：
$$\text{MIG} = \frac{1}{K}\sum_{k=1}^K \frac{I(z_j^*; v_k) - I(z_j^{**}; v_k)}{H(v_k)}$$

其中$v_k$是第$k$个生成因子，$z_j^*$和$z_j^{**}$是互信息最高和次高的隐变量。

**Rule of Thumb**：$\beta$的选择依赖于任务。对于需要可解释表示的任务，通常$\beta \in [4, 10]$；对于纯生成任务，$\beta \in [1, 2]$。

## 10.6 后验崩塌问题

### 10.6.1 问题描述

后验崩塌是VAE训练中的常见问题，表现为：
- 近似后验退化为先验：$q_\phi(z|x) \approx p(z)$
- 隐变量不携带信息：$I(x; z) \approx 0$
- 解码器忽略隐变量，退化为无条件生成器

### 10.6.2 问题的根源

从优化角度分析，当解码器足够强大时，可能出现局部最优：

$$\text{KL}[q_\phi(z|x) \| p(z)] = 0 \Rightarrow q_\phi(z|x) = p(z)$$

此时ELBO简化为：
$$\text{ELBO} = \mathbb{E}_{p(z)}[\log p_\theta(x|z)]$$

### 10.6.3 解决方案

**1. KL退火（KL Annealing）**：
渐进式增加KL项的权重：
$$\mathcal{L}_t = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \lambda_t \cdot \text{KL}[q_\phi(z|x) \| p(z)]$$

其中$\lambda_t$从0逐渐增加到1。

**2. 自由比特（Free Bits）**：
为每个隐变量维度设置最小信息量：
$$\mathcal{L}_{FB} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \sum_i \max(\lambda, \text{KL}_i)$$

**3. 降低解码器容量**：
使用较浅的解码器网络，迫使模型利用隐变量。

**Rule of Thumb**：对于文本VAE，后验崩塌尤其严重，建议结合多种技术；对于图像VAE，通常KL退火就足够。

## 10.7 VAE的变体与扩展

### 10.7.1 条件VAE（CVAE）

引入条件信息$c$（如类别标签）：

$$q_\phi(z|x,c), \quad p_\theta(x|z,c)$$

应用场景：
- 可控生成：指定属性生成样本
- 半监督学习：利用少量标签改善表示

### 10.7.2 层次VAE（Hierarchical VAE）

使用多层隐变量构建更丰富的表示：

```
x ← p(x|z₁) ← z₁ ← p(z₁|z₂) ← z₂ ← p(z₂)
```

优势：
- 捕获多尺度特征
- 更灵活的先验分布
- 改善后验崩塌问题

### 10.7.3 向量量化VAE（VQ-VAE）

使用离散隐表示和码本：

$$z_q = \arg\min_{e_k \in \mathcal{E}} \|z_e - e_k\|_2$$

其中$\mathcal{E} = \{e_1, ..., e_K\}$是可学习的码本。

特点：
- 离散隐空间便于建模
- 避免后验崩塌
- 可与自回归模型结合

## 10.8 历史人物：Diederik P. Kingma与VAE的诞生

### 10.8.1 学术背景

Diederik P. Kingma在阿姆斯特丹大学攻读博士期间，专注于概率模型和深度学习的结合。2013年，他与导师Max Welling共同提出了变分自编码器，几乎同时，Danilo Rezende等人独立提出了类似的想法。

### 10.8.2 关键贡献

**1. 重参数化技巧的创新**

Kingma最重要的贡献是重参数化技巧，巧妙地解决了随机节点的梯度传播问题。这个看似简单的技巧实际上打开了随机计算图优化的大门。

**2. Adam优化器**

同年，Kingma还与Jimmy Ba共同提出了Adam优化器，成为深度学习中最流行的优化算法之一。有趣的是，Adam的名字来源于"Adaptive Moment Estimation"，也暗含"从头开始"的寓意。

### 10.8.3 VAE论文的影响

原始VAE论文"Auto-Encoding Variational Bayes"的特点：
- 简洁优雅的数学推导
- 统一了深度学习与贝叶斯推断
- 启发了大量后续研究

**历史趣事**：VAE论文最初投稿ICLR 2014时是作为workshop论文，但因其重要性后来被提升为会议论文，成为ICLR历史上被引用最多的论文之一。

### 10.8.4 后续发展

Kingma继续在生成模型领域做出贡献：
- **Glow**：基于流的生成模型
- **变分扩散模型**：VAE视角下的扩散模型理解
- 创立公司将生成模型应用于实际问题

**名言**："The best way to understand something is to try to change it." - Kingma经常引用Kurt Lewin的这句话，体现了他通过生成来理解数据的哲学。

## 10.9 现代连接：VAE在大语言模型时代的应用

### 10.9.1 LLM嵌入空间的压缩

现代LLM的嵌入维度通常很高（如4096维），VAE可用于学习低维表示：

**应用场景**：
1. **语义搜索加速**：将高维嵌入压缩到低维空间，加快检索速度
2. **嵌入量化**：结合VQ-VAE实现离散化存储
3. **跨模型对齐**：学习不同LLM嵌入空间的共同表示

**实践案例**：
```
原始嵌入（4096维） → VAE编码器 → 压缩表示（128维）
                                    ↓
                              语义搜索/聚类
```

压缩比可达32:1，同时保留95%以上的语义信息。

### 10.9.2 可控文本生成

**VAE-LLM混合架构**：

1. **隐变量注入**：
   ```
   文本 → LLM编码器 → VAE → z → LLM解码器 → 生成文本
                              ↑
                          控制信号
   ```

2. **风格迁移**：
   - 内容编码器：提取语义信息
   - 风格编码器：提取风格特征
   - 交叉重组实现风格迁移

### 10.9.3 LLM的不确定性量化

VAE提供了量化LLM不确定性的原则性方法：

**后验方差作为不确定性度量**：
$$\text{不确定性} = \text{Tr}(\Sigma_\phi(x))$$

应用：
- 识别模型不确定的预测
- 主动学习中的样本选择
- 幻觉检测

### 10.9.4 多模态理解中的桥梁作用

VAE在连接不同模态中发挥重要作用：

```
图像 → 视觉编码器 ↘
                    VAE共享隐空间 → 多模态表示
文本 → 文本编码器 ↗
```

**CLIP-VAE架构**：
- 利用CLIP的对齐特性
- VAE学习模态不变的隐表示
- 实现零样本跨模态生成

### 10.9.5 计算效率优化

**稀疏VAE for LLM**：

通过学习稀疏隐表示减少计算：
$$\mathcal{L}_{sparse} = \mathcal{L}_{VAE} + \lambda \|z\|_1$$

效果：
- 推理速度提升2-3倍
- 内存占用减少50%
- 保持生成质量

**Rule of Thumb**：在LLM应用中，VAE的隐维度通常设为原始维度的1/8到1/32，平衡压缩率和信息保留。

## 10.10 本章小结

### 核心概念回顾

1. **变分推断框架**：
   - 用可处理的分布$q_\phi(z|x)$近似难解的后验$p(z|x)$
   - ELBO作为log似然的下界，同时也是优化目标

2. **ELBO的两种理解**：
   - 信息论视角：重构项 - KL正则项
   - 期望最大化视角：期望步骤的下界

3. **重参数化技巧**：
   - 核心创新：$z = \mu + \sigma \odot \epsilon$
   - 使随机采样可微，实现端到端训练

4. **关键问题与解决方案**：
   - 后验崩塌：KL退火、自由比特、架构设计
   - 解耦表示：β-VAE通过调整KL权重
   - 离散隐变量：VQ-VAE使用向量量化

### 重要公式总结

1. **ELBO**：
   $$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}[q_\phi(z|x) \| p(z)]$$

2. **高斯VAE的KL散度**（闭式解）：
   $$\text{KL} = \frac{1}{2}\sum_{i=1}^d \left(\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1\right)$$

3. **β-VAE目标**：
   $$\mathcal{L}_{\beta} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - \beta \cdot \text{KL}[q_\phi \| p]$$

### 实用建议

1. **架构设计**：
   - 编码器输出均值和log方差
   - 解码器根据数据类型选择合适的分布
   - 隐维度通常为输入维度的1/10到1/100

2. **训练技巧**：
   - 使用KL退火避免后验崩塌
   - 监控重构项和KL项的平衡
   - 批次大小影响KL估计的方差

3. **超参数选择**：
   - β值：生成质量优先选1-2，解耦优先选4-10
   - 隐维度：过小信息损失，过大难以正则化
   - 学习率：编码器和解码器可使用不同学习率

## 10.11 常见陷阱与错误

### 陷阱1：忽视数值稳定性

**问题**：直接输出方差而非log方差
```
错误：σ² = network(x)  # 可能为负或过大
正确：log_var = network(x); σ² = exp(log_var)
```

**解决**：始终使用log方差，并在计算时加入小常数避免数值问题

### 陷阱2：KL权重设置不当

**问题**：β设置过大导致信息瓶颈
- 症状：重构质量极差，隐变量几乎不变
- 诊断：监控KL项，如果接近0说明后验崩塌

**解决**：使用退火策略或自适应β

### 陷阱3：批次统计的误用

**问题**：在计算KL时使用批次统计而非样本统计
```
错误：kl = mean(kl_per_dim)  # 跨维度平均
正确：kl = sum(kl_per_dim)   # 跨维度求和
```

### 陷阱4：先验选择不当

**问题**：盲目使用标准高斯先验
- 对于某些数据，其他先验可能更合适
- 如混合高斯、或学习的先验

**解决**：根据数据特性选择或学习先验

### 陷阱5：评估指标的误解

**问题**：仅关注重构误差
- VAE的目标不仅是重构，还包括学习良好的隐表示
- 需要综合评估生成质量、插值平滑性、解耦程度

**解决**：使用多个指标：FID、IS、MIG等

### 陷阱6：采样数量不足

**问题**：训练时每个数据点只采样一个z
- 导致梯度估计方差大
- 特别是在隐维度较高时

**解决**：使用重要性加权或多次采样

## 10.12 练习题

### 基础题

**习题10.1** 推导ELBO
证明对于任意分布$q(z|x)$，都有：
$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x,z) - \log q(z|x)]$$

*提示：使用Jensen不等式或KL散度的非负性*

<details>
<summary>答案</summary>

方法一（Jensen不等式）：
$$\log p(x) = \log \int p(x,z)dz = \log \int q(z|x)\frac{p(x,z)}{q(z|x)}dz$$
$$\geq \int q(z|x)\log\frac{p(x,z)}{q(z|x)}dz = \text{ELBO}$$

方法二（KL散度）：
$$\text{KL}[q(z|x)\|p(z|x)] = \int q(z|x)\log\frac{q(z|x)}{p(z|x)}dz \geq 0$$
$$= \int q(z|x)\log q(z|x)dz - \int q(z|x)\log p(z|x)dz$$
$$= \int q(z|x)\log q(z|x)dz - \int q(z|x)\log\frac{p(x,z)}{p(x)}dz$$
$$= -\text{ELBO} + \log p(x)$$

因此$\log p(x) \geq \text{ELBO}$。
</details>

**习题10.2** 高斯KL散度计算
对于两个高斯分布$q = \mathcal{N}(\mu_1, \sigma_1^2)$和$p = \mathcal{N}(\mu_2, \sigma_2^2)$，推导KL散度的闭式解。

*提示：利用高斯分布的熵和交叉熵*

<details>
<summary>答案</summary>

KL散度定义：
$$\text{KL}[q\|p] = \int q(x)\log\frac{q(x)}{p(x)}dx$$

对于高斯分布：
$$\text{KL}[q\|p] = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

当$p = \mathcal{N}(0,1)$时（VAE的常见情况）：
$$\text{KL} = \frac{1}{2}(\mu_1^2 + \sigma_1^2 - \log\sigma_1^2 - 1)$$
</details>

**习题10.3** 重参数化实现
解释为什么直接从$\mathcal{N}(\mu, \sigma^2)$采样无法反向传播，而重参数化$z = \mu + \sigma\epsilon$（其中$\epsilon \sim \mathcal{N}(0,1)$）可以。

*提示：考虑计算图中的随机节点*

<details>
<summary>答案</summary>

直接采样时，$z \sim \mathcal{N}(\mu, \sigma^2)$是一个随机操作，梯度无法通过随机采样操作传播回$\mu$和$\sigma$。

重参数化后：
- $\epsilon$的随机性与参数$\mu, \sigma$无关
- $z = \mu + \sigma\epsilon$是确定性操作
- 梯度可以通过确定性操作传播：
  $$\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon$$

这样，随机性被"外部化"，梯度可以正常流动。
</details>

**习题10.4** β-VAE的信息瓶颈
说明当β→∞时，β-VAE会发生什么？这与信息瓶颈原理有何联系？

*提示：考虑极限情况下的优化目标*

<details>
<summary>答案</summary>

当β→∞时，损失函数被KL项主导：
$$\mathcal{L} \approx -β \cdot \text{KL}[q_\phi(z|x) \| p(z)]$$

为最小化损失，模型会让$q_\phi(z|x) \approx p(z)$，即后验接近先验，隐变量不再携带关于x的信息。

这对应信息瓶颈的极端情况：
- 互信息$I(X; Z) \to 0$
- 形成最严格的信息瓶颈
- 只保留最关键的信息（如果有的话）

实践中，需要在信息保留和压缩之间找平衡。
</details>

### 挑战题

**习题10.5** 条件VAE的ELBO
推导条件VAE（CVAE）的ELBO，其中需要建模$p(x|c)$，c是条件信息。

*提示：在所有分布中加入条件c*

<details>
<summary>答案</summary>

条件VAE的生成过程：
1. 给定条件c
2. 从先验采样：$z \sim p(z|c)$
3. 生成数据：$x \sim p_\theta(x|z,c)$

ELBO推导：
$$\log p(x|c) \geq \mathbb{E}_{q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - \text{KL}[q_\phi(z|x,c) \| p(z|c)]$$

关键区别：
- 编码器：$q_\phi(z|x,c)$接收x和c
- 解码器：$p_\theta(x|z,c)$基于z和c生成
- 先验：可以是条件先验$p(z|c)$或固定先验$p(z)$

应用：可控生成、半监督学习、多任务学习。
</details>

**习题10.6** VQ-VAE的梯度估计
VQ-VAE使用向量量化，这是不可微操作。解释straight-through估计器如何解决这个问题。

*提示：考虑前向传播和反向传播的不同处理*

<details>
<summary>答案</summary>

VQ-VAE的向量量化操作：
$$z_q = \arg\min_{e_k} \|z_e - e_k\|_2$$

这是离散选择，不可微。Straight-through估计器的策略：

**前向传播**：使用量化后的值
$$z_q = e_{k^*}, \quad k^* = \arg\min_k \|z_e - e_k\|_2$$

**反向传播**：将梯度直接传递
$$\frac{\partial \mathcal{L}}{\partial z_e} := \frac{\partial \mathcal{L}}{\partial z_q}$$

额外的损失项：
1. **承诺损失**：$\|z_e - \text{sg}[z_q]\|_2^2$ - 让编码器输出接近码本
2. **码本损失**：$\|\text{sg}[z_e] - z_q\|_2^2$ - 更新码本向量

其中sg表示stop gradient。这种设计巧妙地实现了离散表示的端到端学习。
</details>

**习题10.7** 层次VAE的ELBO
对于两层VAE：$p(x|z_1)p(z_1|z_2)p(z_2)$，推导其ELBO并解释每一项的含义。

*提示：逐层应用变分推断*

<details>
<summary>答案</summary>

两层VAE的ELBO：
$$\log p(x) \geq \mathbb{E}_{q(z_1,z_2|x)}[\log p(x,z_1,z_2) - \log q(z_1,z_2|x)]$$

假设后验分解为：$q(z_1,z_2|x) = q(z_1|x)q(z_2|z_1,x)$

展开得到：
$$\text{ELBO} = \mathbb{E}_{q(z_1|x)}[\log p(x|z_1)] - \text{KL}[q(z_1|x) \| p(z_1|z_2)]$$
$$- \mathbb{E}_{q(z_1|x)}[\text{KL}[q(z_2|z_1,x) \| p(z_2)]]$$

含义：
- 第一项：重构损失
- 第二项：第一层隐变量的正则化（条件先验）
- 第三项：第二层隐变量的正则化

优势：更灵活的先验$p(z_1|z_2)$可以更好地匹配数据分布。
</details>

**习题10.8** 开放思考：VAE与扩散模型的联系
扩散模型可以看作是层次VAE的极限情况。请解释这种联系，并讨论两者的优劣。

*提示：考虑扩散模型的马尔可夫链结构*

<details>
<summary>答案</summary>

**联系**：
扩散模型的前向过程：
$$q(x_1,...,x_T|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$$

可以看作T层的VAE，每层添加少量噪声。

**ELBO的相似性**：
扩散模型的目标也是最大化ELBO：
$$\text{ELBO} = \mathbb{E}_q[\log p(x_0|x_1)] - \sum_{t=2}^T \text{KL}[q(x_{t-1}|x_t,x_0) \| p(x_{t-1}|x_t)]$$

**关键区别**：
1. **层数**：VAE通常2-3层，扩散模型可达1000层
2. **隐变量维度**：VAE降维，扩散模型保持维度
3. **先验设计**：VAE学习编码，扩散模型固定加噪过程

**优劣比较**：
- VAE：可解释的隐空间，快速采样，但生成质量受限
- 扩散模型：高质量生成，但采样慢，隐空间不可解释

**统一视角**：两者都是通过分层隐变量建模复杂分布，权衡了表达能力和计算效率。
</details>

---

**继续学习**：[第11章：扩散模型](chapter11.md) →