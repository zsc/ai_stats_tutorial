# 第11章：扩散模型

扩散模型（Diffusion Models）代表了生成模型的最新突破，在图像生成、音频合成、分子设计等领域取得了前所未有的成功。本章将从统计学和优化理论的角度深入理解扩散模型的原理，掌握其训练和采样技术，并探索在实际应用中的关键技巧。

## 11.1 引言与动机

### 生成模型的演进

生成模型的目标是学习数据分布 $p_{\text{data}}(\mathbf{x})$，从而能够生成新的样本。在扩散模型出现之前，主要的生成模型包括：

1. **VAE（变分自编码器）**：通过变分推断学习潜在表示，生成质量受限于高斯假设
2. **GAN（生成对抗网络）**：通过对抗训练获得高质量生成，但训练不稳定且模式覆盖不全
3. **自回归模型**：逐元素生成，计算代价高且难以并行化
4. **流模型**：通过可逆变换建模，但架构设计受限

扩散模型巧妙地结合了这些方法的优点：
- 像VAE一样有稳定的训练目标
- 像GAN一样能生成高质量样本
- 像自回归模型一样有良好的似然估计
- 像流模型一样有理论保证

### 核心思想：破坏与重建

扩散模型的核心思想极其简单而优雅：

```
原始数据 → [逐步加噪] → 纯噪声
纯噪声 → [逐步去噪] → 生成数据
```

这个过程类似于：
1. **物理扩散**：墨水滴入水中逐渐扩散，最终均匀分布
2. **信息论视角**：逐步增加熵直到最大熵状态
3. **统计视角**：从数据分布逐步过渡到先验分布（通常是高斯分布）

关键洞察是：如果我们能学习反向过程（去噪），就能从噪声生成数据。

## 11.2 前向扩散过程

### 马尔可夫链定义

前向扩散过程定义为一个马尔可夫链，逐步向数据添加高斯噪声：

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

其中：
- $\mathbf{x}_0 \sim p_{\text{data}}$ 是原始数据
- $\mathbf{x}_1, \ldots, \mathbf{x}_T$ 是逐步加噪的中间状态
- $\beta_t \in (0, 1)$ 是噪声调度（noise schedule）
- $T$ 是扩散步数（通常为1000）

这个设计有几个关键考虑：
1. **方差保持性**：选择 $\sqrt{1-\beta_t}$ 作为均值系数确保在适当条件下方差不会爆炸
2. **渐进性**：$\beta_t$ 通常很小（$10^{-4}$ 到 $0.02$），确保缓慢破坏结构
3. **可逆性**：高斯噪声的选择使得理论分析tractable

### 重参数化技巧与闭式解

虽然前向过程是马尔可夫的，但我们可以直接从 $\mathbf{x}_0$ 采样任意时刻 $\mathbf{x}_t$：

定义 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$，则：

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})$$

**推导过程**：利用高斯分布的可加性
$$\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1-\alpha_t} \boldsymbol{\epsilon}_{t-1}$$
$$= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}} \boldsymbol{\epsilon}_{t-2}) + \sqrt{1-\alpha_t} \boldsymbol{\epsilon}_{t-1}$$
$$= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \text{组合噪声项}$$

递归展开并利用独立高斯噪声的可加性，最终得到：
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$$

其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$。

**直觉理解**：
- $\sqrt{\bar{\alpha}_t}$ 控制保留多少原始信号（信号衰减因子）
- $\sqrt{1-\bar{\alpha}_t}$ 控制添加多少噪声（噪声放大因子）
- 当 $t \to T$，$\bar{\alpha}_T \to 0$，$\mathbf{x}_T$ 接近纯高斯噪声
- 系数平方和为1：$\bar{\alpha}_t + (1-\bar{\alpha}_t) = 1$（能量守恒）

这个闭式解的重要性：
1. **训练效率**：可以直接采样任意时刻，无需逐步模拟
2. **并行化**：不同时刻的样本可以并行生成
3. **数值稳定**：避免累积误差

### 噪声调度设计

噪声调度 $\{\beta_t\}_{t=1}^T$ 的设计至关重要，直接影响模型的训练效率和生成质量：

1. **线性调度**：$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$
   - 简单直观，早期工作常用
   - 典型值：$\beta_{\min} = 10^{-4}$，$\beta_{\max} = 0.02$
   - 问题：早期噪声添加过快，后期过慢

2. **余弦调度**：基于信噪比设计
   $$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$
   - 更平滑的噪声添加过程
   - 在中间时间步保留更多信息
   - 参数 $s=0.008$ 防止 $t=0$ 附近的突变

3. **平方根调度**：$\bar{\alpha}_t = 1 - \sqrt{t/T + s}$
   - 介于线性和余弦之间
   - 对高分辨率图像效果较好

4. **自适应调度**：根据数据特性学习最优调度
   - 可以用神经网络参数化 $\beta_t$
   - 需要额外的正则化防止退化

**调度设计原则**：
- 初始阶段（$t$ 小）：保持大部分信息，$\beta_t$ 应该很小
- 中间阶段：平稳过渡，避免信息丢失过快
- 最终阶段（$t$ 接近 $T$）：确保 $\mathbf{x}_T \approx \mathcal{N}(0, \mathbf{I})$

**Rule of Thumb**：
- 图像生成：余弦调度通常更好，特别是256×256以下分辨率
- 高分辨率（512×512以上）：考虑平方根或学习的调度
- 步数选择：训练时 $T=1000$，推理时可以减少到 50-100 步
- 调试技巧：可视化不同 $t$ 时刻的 $\mathbf{x}_t$，确保噪声添加合理
- 检查 $\bar{\alpha}_T$ 是否足够小（通常 < 0.001）

### 信噪比分析

信噪比（SNR）提供了理解扩散过程的统一框架：

定义信噪比：
$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t} = \frac{\text{信号方差}}{\text{噪声方差}}$$

对数信噪比：
$$\log \text{SNR}(t) = \log \bar{\alpha}_t - \log(1-\bar{\alpha}_t)$$

**关键性质**：
1. **单调性**：SNR 严格单调递减
   - $\text{SNR}(0) = \infty$（纯信号）
   - $\text{SNR}(T) \approx 0$（纯噪声）

2. **调度等价性**：不同调度可以通过SNR匹配实现等价
   - 给定目标 $\log \text{SNR}(t)$，可以反推 $\bar{\alpha}_t$
   - $\bar{\alpha}_t = \text{sigmoid}(\log \text{SNR}(t))$

3. **最优调度特征**：
   - 均匀的对数SNR下降通常导致更好的生成质量
   - $\frac{d \log \text{SNR}}{dt} \approx \text{const}$ 是理想情况

4. **分辨率适应**：
   - 高分辨率图像需要更缓慢的SNR下降
   - 可以通过SNR匹配在不同分辨率间迁移模型

**实用技巧**：
- 监控训练时不同 $t$ 的损失分布，理想情况下应该均匀
- 如果某些 $t$ 的损失特别高，考虑调整该区间的噪声调度
- 使用SNR加权的损失函数可以改善训练稳定性

## 11.3 反向扩散与去噪

### 反向过程的参数化

反向扩散过程也是一个马尔可夫链，但需要学习：

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

关键问题：如何参数化均值 $\boldsymbol{\mu}_\theta$ 和方差 $\boldsymbol{\Sigma}_\theta$？

**理论基础**：当 $\beta_t$ 足够小时，反向过程的真实分布也近似高斯：
$$q(\mathbf{x}_{t-1}|\mathbf{x}_t) \approx \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_q(\mathbf{x}_t, t), \boldsymbol{\Sigma}_q(t))$$

这为高斯参数化提供了理论支撑。

### 贝叶斯后验的启发

当我们知道 $\mathbf{x}_0$ 时，可以利用贝叶斯定理计算真实的后验分布：

$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$$

**推导**：利用贝叶斯定理和高斯分布的性质
$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)}$$

由于马尔可夫性质：$q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t | \mathbf{x}_{t-1})$

代入高斯分布形式并完成平方，得到：
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t$$

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$

**关键洞察**：
1. 后验均值是 $\mathbf{x}_0$ 和 $\mathbf{x}_t$ 的线性组合
2. 权重系数依赖于噪声调度
3. 后验方差是确定的，不依赖于数据

这启发我们：如果能从 $\mathbf{x}_t$ 预测出 $\mathbf{x}_0$，就能计算反向过程的均值！

### 三种等价参数化

给定 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$，我们可以选择预测：

1. **预测原始数据** ($\mathbf{x}_0$-参数化)：$\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_t, t)$
   
   从噪声样本重构原始数据：
   $$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$
   
   代入后验均值公式：
   $$\boldsymbol{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \hat{\mathbf{x}}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t$$

2. **预测噪声** ($\boldsymbol{\epsilon}$-参数化)：$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
   
   这是最直观的选择，因为前向过程就是添加噪声：
   $$\boldsymbol{\mu}_\theta = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\boldsymbol{\epsilon}} \right)$$
   
   训练目标简化为：$\mathcal{L} = \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$

3. **预测速度** (v-参数化)：$\hat{v} = v_\theta(\mathbf{x}_t, t)$
   
   定义速度：$v = \sqrt{\bar{\alpha}_t} \boldsymbol{\epsilon} - \sqrt{1-\bar{\alpha}_t} \mathbf{x}_0$
   
   这种参数化在不同SNR区域都稳定：
   - 高SNR时，$v \approx -\mathbf{x}_0$（主要预测数据）
   - 低SNR时，$v \approx \boldsymbol{\epsilon}$（主要预测噪声）

**参数化之间的转换**：
- 从噪声到数据：$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \hat{\boldsymbol{\epsilon}}}{\sqrt{\bar{\alpha}_t}}$
- 从数据到噪声：$\hat{\boldsymbol{\epsilon}} = \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \hat{\mathbf{x}}_0}{\sqrt{1-\bar{\alpha}_t}}$
- 从速度到数据/噪声：$\hat{\mathbf{x}}_0 = \sqrt{\bar{\alpha}_t} \mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \hat{v}$

**实践中的选择**：
- **预测噪声**：DDPM默认，训练稳定，适合大多数场景
- **预测数据**：在低SNR区域（$t$ 接近 $T$）更稳定，适合高分辨率
- **v-参数化**：结合两者优点，Progressive Distillation等方法常用
- **混合策略**：根据 $t$ 动态选择参数化

### 去噪分数匹配视角

扩散模型与分数匹配有深刻联系，这提供了另一个理论框架。

**分数函数**定义为对数概率密度的梯度：
$$\mathbf{s}(\mathbf{x}, t) = \nabla_\mathbf{x} \log p_t(\mathbf{x})$$

分数函数的重要性：
1. 不需要归一化常数（对比能量模型）
2. 可以通过朗之万动力学采样
3. 与去噪有自然联系

**去噪分数匹配目标**：
$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \lambda(t) \left\| \mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) \right\|^2 \right]$$

由于 $q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$ 是高斯分布，其分数有闭式解：

$$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{1}{1-\bar{\alpha}_t}(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}$$

**关键洞察**：学习分数等价于学习去噪！

这建立了三个等价视角：
1. **概率视角**：学习反向转移概率 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$
2. **去噪视角**：学习从噪声数据恢复干净数据
3. **分数视角**：学习数据分布的分数函数

**连续时间极限**：当 $T \to \infty$，扩散过程收敛到随机微分方程（SDE）：
$$d\mathbf{x} = f(\mathbf{x}, t)dt + g(t)d\mathbf{w}$$

对应的概率流ODE：
$$\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})$$

## 11.4 DDPM：去噪扩散概率模型

### 变分下界推导

DDPM通过最大化数据似然的变分下界（ELBO）来训练：

$$\log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_q \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right] = -\mathcal{L}_{\text{VLB}}$$

展开变分下界：
$$\mathcal{L}_{\text{VLB}} = \mathbb{E}_q \left[ \underbrace{D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}} - \underbrace{\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0} \right]$$

其中：
- $L_T$：先验匹配项，通常可忽略（设计使 $q(\mathbf{x}_T|\mathbf{x}_0) \approx p(\mathbf{x}_T) = \mathcal{N}(0, \mathbf{I})$）
- $L_{t-1}$：去噪匹配项，这是主要的训练目标
- $L_0$：重建项，可以用离散化的高斯似然计算

### 简化训练目标

Ho等人(2020)发现，简化的目标函数效果更好：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t \sim \mathcal{U}(1,T), \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right]$$

这个目标：
1. 忽略了权重系数（实验发现影响不大）
2. 直接预测噪声而非分布参数
3. 均匀采样时间步

**训练算法**：
```
输入: 数据集 D, 噪声调度 {βₜ}, 模型 εθ
1. 重复直到收敛:
2.   从 D 中采样 x₀
3.   均匀采样 t ~ U(1, T)
4.   采样噪声 ε ~ N(0, I)
5.   计算 xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε
6.   梯度下降更新: ∇θ ||ε - εθ(xₜ, t)||²
```

### 采样算法

DDPM的采样过程从噪声开始，逐步去噪：

```
输入: 训练好的模型 εθ
1. 采样 xₜ ~ N(0, I)
2. 对于 t = T, T-1, ..., 1:
3.   计算 z ~ N(0, I) 如果 t > 1，否则 z = 0
4.   计算 x_{t-1} = 1/√αₜ (xₜ - βₜ/√(1-ᾱₜ) εθ(xₜ, t)) + σₜ z
其中 σₜ² = β̃ₜ 或 βₜ (两种选择)
```

### 方差选择与性能

反向过程的方差 $\boldsymbol{\Sigma}_\theta$ 有两种常见选择：

1. **固定小方差**：$\boldsymbol{\Sigma}_\theta = \tilde{\beta}_t \mathbf{I} = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t \mathbf{I}$
   - 理论上的后验方差下界
   - DDPM原文选择

2. **固定大方差**：$\boldsymbol{\Sigma}_\theta = \beta_t \mathbf{I}$
   - 更简单的选择
   - 某些情况下效果相当

3. **学习方差**：$\boldsymbol{\Sigma}_\theta = \exp(v_\theta(\mathbf{x}_t, t) \log \beta_t + (1-v_\theta(\mathbf{x}_t, t)) \log \tilde{\beta}_t)$
   - 插值between两个极端
   - 可以改善似然但对样本质量影响有限

**Rule of Thumb**：
- 开始时使用固定小方差
- 如果需要更好的似然估计，再考虑学习方差
- 采样质量主要由均值预测决定

## 11.5 DDIM：去噪扩散隐式模型

### 动机：加速采样

DDPM的主要缺点是采样速度慢，需要T步（通常1000步）迭代。DDIM通过构造非马尔可夫扩散过程实现快速采样。

### 非马尔可夫前向过程

DDIM定义了一族满足边缘分布 $q(\mathbf{x}_t|\mathbf{x}_0)$ 不变的前向过程：

$$q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2 \mathbf{I}\right)$$

当 $\sigma_t = \tilde{\beta}_t$ 时，退化为DDPM；当 $\sigma_t = 0$ 时，过程变为确定性。

### 确定性采样

设置 $\sigma_t = 0$，DDIM的采样变为确定性：

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{预测的} \mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

**关键洞察**：
1. DDIM将采样视为数值ODE求解
2. 可以使用更大的步长（子序列采样）
3. 确定性允许精确反演和插值

### 加速采样策略

DDIM允许使用子序列 $\tau = \{\tau_1, \tau_2, ..., \tau_S\} \subset \{1, 2, ..., T\}$ 进行采样：

```
输入: 模型 εθ, 步数 S << T
1. 构造子序列 τ (如均匀间隔)
2. 采样 x_{τₛ} ~ N(0, I)
3. 对于 i = S, S-1, ..., 1:
4.   计算 x_{τᵢ₋₁} 使用DDIM更新规则
     从 x_{τᵢ} 直接跳到 x_{τᵢ₋₁}
```

**采样步数选择**：
- DDPM: 1000步
- DDIM: 10-50步常见
- 质量vs速度权衡：20-50步通常足够

### DDIM的独特性质

1. **语义插值**：由于确定性，可以在潜在空间插值
   $$\mathbf{z}_{\text{interp}} = \sqrt{1-\lambda} \mathbf{z}_1 + \sqrt{\lambda} \mathbf{z}_2$$

2. **精确反演**：可以将图像编码回噪声空间
   $$\mathbf{x}_{t+1} = \sqrt{\bar{\alpha}_{t+1}} \hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t+1}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

3. **一致性**：相同初始噪声产生相同结果（利于调试）