# 第3章：线性模型与正则化

线性模型是机器学习的基石，虽然结构简单，却蕴含着深刻的统计学原理。本章将从最小二乘法出发，逐步引入正则化技术，最终以贝叶斯视角统一这些方法。我们将看到，许多现代深度学习中的技术思想，其根源都可以追溯到这些经典的线性方法。

## 3.1 最小二乘法与岭回归

### 3.1.1 最小二乘法的几何与代数

考虑线性回归问题：给定数据集 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$，其中 $\mathbf{x}_i \in \mathbb{R}^d$，$y_i \in \mathbb{R}$，我们希望找到参数 $\mathbf{w} \in \mathbb{R}^d$ 和 $b \in \mathbb{R}$，使得：

$$\hat{y} = \mathbf{w}^T\mathbf{x} + b$$

这个简单的线性假设背后蕴含着深刻的洞察：在局部范围内，大多数非线性关系都可以用线性函数近似（泰勒展开的一阶项）。即使在深度学习时代，线性层仍然是神经网络的基本构建块。

最小二乘法通过最小化平方损失来求解：

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{2n}\sum_{i=1}^n (y_i - \mathbf{w}^T\mathbf{x}_i - b)^2$$

选择平方损失而非绝对值损失有多重原因：
1. **可微性**：平方函数处处可微，便于优化
2. **唯一性**：凸函数保证全局最优解
3. **概率解释**：对应高斯噪声的最大似然估计
4. **计算效率**：导致线性方程组，有闭式解

为简化记号，我们将偏置项并入权重向量，记 $\tilde{\mathbf{x}} = [\mathbf{x}^T, 1]^T$，$\tilde{\mathbf{w}} = [\mathbf{w}^T, b]^T$。令 $\mathbf{X} \in \mathbb{R}^{n \times (d+1)}$ 为设计矩阵，$\mathbf{y} \in \mathbb{R}^n$ 为目标向量，则：

$$\mathcal{L}(\tilde{\mathbf{w}}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}\|_2^2$$

展开这个表达式：
$$\mathcal{L}(\tilde{\mathbf{w}}) = \frac{1}{2n}(\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}})^T(\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}) = \frac{1}{2n}(\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\tilde{\mathbf{w}} + \tilde{\mathbf{w}}^T\mathbf{X}^T\mathbf{X}\tilde{\mathbf{w}})$$

**正规方程的推导**：对 $\tilde{\mathbf{w}}$ 求导并令其为零：

$$\nabla_{\tilde{\mathbf{w}}} \mathcal{L} = \frac{1}{n}(-\mathbf{X}^T\mathbf{y} + \mathbf{X}^T\mathbf{X}\tilde{\mathbf{w}}) = 0$$

这给出正规方程：
$$\mathbf{X}^T\mathbf{X}\tilde{\mathbf{w}} = \mathbf{X}^T\mathbf{y}$$

当 $\mathbf{X}^T\mathbf{X}$ 可逆时（满秩条件：$\text{rank}(\mathbf{X}) = d+1 \leq n$），解为：

$$\tilde{\mathbf{w}}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

矩阵 $\mathbf{X}^+ = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ 称为Moore-Penrose伪逆。

**几何解释**：
```
    y
    |\ 
    | \
    |  \ 残差 e = y - ŷ
    |   \
    |    * 数据点
    |   /|
    |  / |
    | /  | 投影
    |/___|_______> Col(X)
      ŷ = Xw*
```

最小二乘解 $\hat{\mathbf{y}} = \mathbf{X}\tilde{\mathbf{w}}^*$ 是 $\mathbf{y}$ 在列空间 $\text{Col}(\mathbf{X})$ 上的正交投影。这个几何视角揭示了几个重要性质：

1. **正交性条件**：残差 $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ 正交于列空间：$\mathbf{X}^T\mathbf{e} = \mathbf{0}$
2. **投影矩阵**：$\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ 是投影算子，满足 $\mathbf{H}^2 = \mathbf{H}$
3. **最短距离**：在所有 $\mathbf{X}\mathbf{w}$ 形式的预测中，$\hat{\mathbf{y}}$ 与 $\mathbf{y}$ 的欧氏距离最小

**统计性质**：

假设数据生成过程为 $\mathbf{y} = \mathbf{X}\mathbf{w}_{\text{true}} + \boldsymbol{\epsilon}$，其中 $\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$，$\text{Var}[\boldsymbol{\epsilon}] = \sigma^2\mathbf{I}$，则：

1. **无偏性**：$\mathbb{E}[\hat{\mathbf{w}}] = \mathbf{w}_{\text{true}}$
2. **方差**：$\text{Var}[\hat{\mathbf{w}}] = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$
3. **Gauss-Markov定理**：在所有线性无偏估计中，OLS具有最小方差

### 3.1.2 病态问题与岭回归

当特征高度相关或 $d > n$ 时，$\mathbf{X}^T\mathbf{X}$ 接近奇异，导致严重的数值和统计问题。

**病态性的表现**：
1. **数值不稳定**：微小扰动引起解的巨大变化
2. **过拟合**：模型在训练集上表现良好，但泛化性能差  
3. **系数爆炸**：参数估计值异常大，符号可能错误
4. **方差膨胀**：参数估计的方差趋于无穷

**条件数分析**：

矩阵条件数定义为：
$$\kappa(\mathbf{X}^T\mathbf{X}) = \frac{\lambda_{\max}(\mathbf{X}^T\mathbf{X})}{\lambda_{\min}(\mathbf{X}^T\mathbf{X})} = \left(\frac{\sigma_{\max}(\mathbf{X})}{\sigma_{\min}(\mathbf{X})}\right)^2$$

其中 $\sigma_i$ 是 $\mathbf{X}$ 的奇异值。条件数的含义：
- $\kappa < 10$：良态问题
- $10 < \kappa < 100$：轻度病态
- $100 < \kappa < 1000$：中度病态
- $\kappa > 1000$：严重病态

**相对误差放大**：若输入有相对误差 $\epsilon$，则解的相对误差可达 $\kappa \cdot \epsilon$。

**岭回归（Ridge Regression）** 通过添加 $L_2$ 正则化来缓解这些问题：

$$\mathcal{L}_{\text{ridge}}(\tilde{\mathbf{w}}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}\|_2^2 + \frac{\lambda}{2}\|\tilde{\mathbf{w}}\|_2^2$$

其中 $\lambda > 0$ 是正则化参数，控制偏差-方差权衡。

**闭式解的推导**：

令梯度为零：
$$\nabla_{\tilde{\mathbf{w}}} \mathcal{L}_{\text{ridge}} = \frac{1}{n}\mathbf{X}^T(\mathbf{X}\tilde{\mathbf{w}} - \mathbf{y}) + \lambda\tilde{\mathbf{w}} = 0$$

整理得：
$$(\mathbf{X}^T\mathbf{X} + n\lambda\mathbf{I})\tilde{\mathbf{w}} = \mathbf{X}^T\mathbf{y}$$

因此：
$$\tilde{\mathbf{w}}_{\text{ridge}}^* = (\mathbf{X}^T\mathbf{X} + n\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

**改善条件数**：正则化后的条件数变为：
$$\kappa_{\text{ridge}} = \frac{\lambda_{\max} + n\lambda}{\lambda_{\min} + n\lambda} < \kappa$$

当 $\lambda$ 足够大时，$\kappa_{\text{ridge}} \approx 1$，问题变为良态。

**Rule of thumb**：
- 特征相关性高：$\lambda \in [0.1, 1.0]$
- 样本量充足（$n \gg d$）：$\lambda \in [0.001, 0.01]$
- 高维问题（$d > n$）：$\lambda \in [1.0, 10.0]$
- 实践中：使用交叉验证，在对数尺度 $[10^{-4}, 10^2]$ 上搜索

### 3.1.3 奇异值分解视角

通过SVD分解 $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$，我们可以更深入理解岭回归的作用机制。这里：
- $\mathbf{U} \in \mathbb{R}^{n \times n}$：左奇异向量，数据空间的正交基
- $\mathbf{\Sigma} \in \mathbb{R}^{n \times (d+1)}$：奇异值对角矩阵
- $\mathbf{V} \in \mathbb{R}^{(d+1) \times (d+1)}$：右奇异向量，参数空间的正交基

**最小二乘解的SVD表示**：

$$\tilde{\mathbf{w}}_{\text{OLS}}^* = \mathbf{V}\mathbf{\Sigma}^+\mathbf{U}^T\mathbf{y} = \sum_{i=1}^r \frac{1}{\sigma_i}(\mathbf{u}_i^T\mathbf{y})\mathbf{v}_i$$

其中 $r = \text{rank}(\mathbf{X})$，$\mathbf{\Sigma}^+$ 是伪逆。

**岭回归解的SVD表示**：

$$\tilde{\mathbf{w}}_{\text{ridge}}^* = \mathbf{V}\mathbf{D}_\lambda\mathbf{U}^T\mathbf{y} = \sum_{i=1}^r \frac{\sigma_i}{\sigma_i^2 + n\lambda}(\mathbf{u}_i^T\mathbf{y})\mathbf{v}_i$$

其中 $\mathbf{D}_\lambda = \text{diag}\left(\frac{\sigma_i}{\sigma_i^2 + n\lambda}\right)$。

**收缩因子分析**：

定义收缩因子 $s_i(\lambda) = \frac{\sigma_i^2}{\sigma_i^2 + n\lambda}$，则：
- 当 $\sigma_i \gg \sqrt{n\lambda}$：$s_i \approx 1$（几乎不收缩）
- 当 $\sigma_i \ll \sqrt{n\lambda}$：$s_i \approx 0$（强烈收缩）
- 当 $\sigma_i = \sqrt{n\lambda}$：$s_i = 0.5$（半收缩点）

这揭示了岭回归的**频谱滤波**性质：
```
收缩因子
  1.0 |******
      |      ****
  0.5 |          **
      |            ***
  0.0 |_______________****___
      0   σ₁  σ₂  σ₃ ... σᵣ
         大←奇异值→小
```

**方差缩减效应**：

参数估计的协方差矩阵：
$$\text{Cov}[\tilde{\mathbf{w}}_{\text{ridge}}] = \sigma^2\mathbf{V}\text{diag}\left(\frac{\sigma_i^2}{(\sigma_i^2 + n\lambda)^2}\right)\mathbf{V}^T$$

总方差（迹）：
$$\text{tr}(\text{Cov}[\tilde{\mathbf{w}}_{\text{ridge}}]) = \sigma^2\sum_{i=1}^r \frac{\sigma_i^2}{(\sigma_i^2 + n\lambda)^2} < \sigma^2\sum_{i=1}^r \frac{1}{\sigma_i^2}$$

岭回归通过牺牲无偏性换取方差的大幅降低。

## 3.2 LASSO与稀疏性

### 3.2.1 $L_1$ 正则化的稀疏诱导性

LASSO（Least Absolute Shrinkage and Selection Operator）由Tibshirani于1996年提出，使用 $L_1$ 正则化实现自动特征选择：

$$\mathcal{L}_{\text{lasso}}(\tilde{\mathbf{w}}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}\|_2^2 + \lambda\|\tilde{\mathbf{w}}\|_1$$

其中 $\|\tilde{\mathbf{w}}\|_1 = \sum_{j=1}^{d+1}|w_j|$ 是 $L_1$ 范数。

**为什么 $L_1$ 产生稀疏性？**

从三个角度理解：

**1. 几何视角**：
```
    w2
     ^
     |     岭回归约束域
     |    /----------\
     |   /   圆形    \
     |  |      *最优解|
     |   \   (内部)  /
     |    \--------/
 ----|----+------+----> w1
     |   /|\    /|\
     |  / | \  / | \
     | /  |  \/  |  \  LASSO约束域
     |<---|---*---|---> （菱形）
     |    |  最优解
     |    | (在顶点→稀疏)
```

等高线与约束域的切点：
- $L_2$ 球：切点几乎总在内部，所有坐标非零
- $L_1$ 菱形：切点常在顶点或边上，某些坐标为零

**2. 次梯度视角**：

LASSO的最优性条件（KKT条件）：
$$\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}) = n\lambda \cdot \text{sign}(\hat{w}_j), \quad \text{if } \hat{w}_j \neq 0$$
$$|\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\hat{\mathbf{w}})| \leq n\lambda, \quad \text{if } \hat{w}_j = 0$$

这意味着特征与残差的相关性必须达到阈值 $n\lambda$ 才能"进入"模型。

**3. 贝叶斯视角**：

$L_1$ 正则化对应拉普拉斯先验：
$$p(w_j) = \frac{\lambda}{2}\exp(-\lambda|w_j|)$$

拉普拉斯分布在零点有尖峰，高概率产生零值，而高斯先验（对应 $L_2$）在零点平滑。

```
概率密度
  ^
  |    拉普拉斯
  |      /\
  |     /  \
  |    /    \___
  |___/________\___> w
      0
  
  |    高斯
  |    ___
  |   /   \
  |  /     \
  |_/_______\___> w
      0
```

### 3.2.2 软阈值算子与坐标下降

虽然LASSO没有闭式解，但可以通过迭代算法高效求解。关键洞察是：固定其他坐标时，单变量子问题有闭式解。

**单变量LASSO问题**：

固定 $\mathbf{w}_{-j}$，关于 $w_j$ 的目标函数：
$$f(w_j) = \frac{1}{2n}\sum_{i=1}^n (r_i - x_{ij}w_j)^2 + \lambda|w_j|$$

其中 $r_i = y_i - \sum_{k \neq j} x_{ik}w_k$ 是部分残差。

**软阈值算子**：

上述问题的解由软阈值算子给出：

$$w_j^* = S_{\lambda/c_j}(\rho_j/c_j)$$

其中：
- $\rho_j = \frac{1}{n}\sum_{i=1}^n x_{ij}r_i$ 是特征与残差的相关性
- $c_j = \frac{1}{n}\sum_{i=1}^n x_{ij}^2$ 是特征的二阶矩
- 软阈值算子定义为：

$$S_{\lambda}(x) = \text{sign}(x)\max(|x| - \lambda, 0) = \begin{cases}
x - \lambda & \text{if } x > \lambda \\
0 & \text{if } |x| \leq \lambda \\
x + \lambda & \text{if } x < -\lambda
\end{cases}$$

**软阈值的图像**：
```
     输出 S_λ(x)
         ^
        /|
       / |
      /  |
    _/   |   \_
----+----+----+----> 输入 x
   -λ    0    λ
    \_   |   _/
      \  |  /
       \ | /
        \|/
```

**坐标下降算法**：
```
输入：数据 (X, y)，正则化参数 λ，容差 ε
初始化：w = 0, r = y (残差)
预计算：c_j = ||X_j||²/n for j = 1,...,d

重复直到收敛：
    for j = 1 to d:
        # 计算特征j与当前残差的相关性
        ρ_j = X_j^T r / n
        
        # 保存旧值
        w_old = w_j
        
        # 软阈值更新
        w_j = S_{λ}(ρ_j/c_j + w_old) 
        
        # 更新残差
        if w_j ≠ w_old:
            r = r - (w_j - w_old) * X_j
    
    # 检查收敛
    if max_j |w_j^{new} - w_j^{old}| < ε:
        break
        
返回：稀疏解 w
```

**收敛性质**：
- 每次更新都减少目标函数（下降性）
- 对凸问题保证收敛到全局最优
- 收敛速度：线性收敛，率依赖于相关性结构

### 3.2.3 变量选择与解路径

LASSO的一个重要特性是自动变量选择。随着 $\lambda$ 从大到小变化：
- $\lambda = \lambda_{\max}$：所有系数为0
- $\lambda$ 减小：系数逐个"进入"模型
- $\lambda = 0$：退化为最小二乘解（如果存在）

**Rule of thumb**：
- 特征选择：使用 $\lambda$ 使得选中特征数约为 $\sqrt{n}$
- 预测精度：通过5-10折交叉验证选择 $\lambda$
- 探索性分析：绘制完整的解路径，观察特征进入顺序

## 3.3 弹性网络

### 3.3.1 结合 $L_1$ 和 $L_2$ 的优势

弹性网络（Elastic Net）结合了岭回归和LASSO：

$$\mathcal{L}_{\text{elastic}}(\tilde{\mathbf{w}}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}\|_2^2 + \lambda(\alpha\|\tilde{\mathbf{w}}\|_1 + \frac{1-\alpha}{2}\|\tilde{\mathbf{w}}\|_2^2)$$

其中 $\alpha \in [0,1]$ 控制 $L_1$ 和 $L_2$ 正则化的相对权重。

**优势**：
- 当特征高度相关时，LASSO倾向于随机选择其中一个，弹性网络倾向于同时选择
- 当 $d \gg n$ 时，LASSO最多选择 $n$ 个特征，弹性网络无此限制
- 继承了LASSO的稀疏性和岭回归的群组效应

### 3.3.2 坐标下降求解

弹性网络的坐标更新规则：

$$w_j^* = \frac{S_{\lambda\alpha}(\rho_j)}{1 + \lambda(1-\alpha)}$$

这可以看作先进行LASSO软阈值，再进行岭回归收缩。

### 3.3.3 参数选择策略

**两阶段策略**：
1. 固定 $\alpha$（如0.5），通过交叉验证选择 $\lambda$
2. 在最优 $\lambda$ 附近，微调 $\alpha$

**Rule of thumb**：
- 特征高度相关：$\alpha = 0.2-0.5$（更多 $L_2$）
- 需要稀疏解：$\alpha = 0.7-0.9$（更多 $L_1$）
- 不确定时：$\alpha = 0.5$ 是合理的默认值

## 3.4 贝叶斯线性回归

### 3.4.1 从频率派到贝叶斯派

贝叶斯方法将参数 $\mathbf{w}$ 视为随机变量，引入先验分布 $p(\mathbf{w})$。

**模型假设**：
- 似然：$y|\mathbf{x}, \mathbf{w} \sim \mathcal{N}(\mathbf{w}^T\mathbf{x}, \sigma^2)$
- 先验：$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$

通过贝叶斯定理，后验分布为：

$$p(\mathbf{w}|\mathcal{D}) \propto p(\mathcal{D}|\mathbf{w})p(\mathbf{w})$$

### 3.4.2 后验分布的解析形式

高斯先验与高斯似然的共轭性使得后验也是高斯分布：

$$p(\mathbf{w}|\mathcal{D}) = \mathcal{N}(\mathbf{m}_N, \mathbf{S}_N)$$

其中：
- 后验均值：$\mathbf{m}_N = \mathbf{S}_N\mathbf{X}^T\mathbf{y}/\sigma^2$
- 后验协方差：$\mathbf{S}_N^{-1} = \mathbf{X}^T\mathbf{X}/\sigma^2 + \mathbf{I}/\tau^2$

**与岭回归的联系**：后验均值 $\mathbf{m}_N$ 恰好是岭回归解，其中 $\lambda = \sigma^2/\tau^2$。

### 3.4.3 预测分布与不确定性量化

对新样本 $\mathbf{x}_*$，预测分布为：

$$p(y_*|\mathbf{x}_*, \mathcal{D}) = \int p(y_*|\mathbf{x}_*, \mathbf{w})p(\mathbf{w}|\mathcal{D})d\mathbf{w}$$

这也是高斯分布：

$$y_*|\mathbf{x}_*, \mathcal{D} \sim \mathcal{N}(\mathbf{m}_N^T\mathbf{x}_*, \sigma^2 + \mathbf{x}_*^T\mathbf{S}_N\mathbf{x}_*)$$

**不确定性分解**：
- 偶然不确定性（aleatoric）：$\sigma^2$，数据固有噪声
- 认知不确定性（epistemic）：$\mathbf{x}_*^T\mathbf{S}_N\mathbf{x}_*$，参数不确定性

```
    预测值
      ^
      |     /---置信区间---\
      |    /    _____      \
      |   /  .-´     `-.    \
      |  /  /  均值预测 \    \
      | /  |      *      |    \
      |/___|_____________|_____\___> x
           认知不确定性
        <---------------->
            总不确定性
```

### 3.4.4 自动相关性确定（ARD）

通过引入不同的先验精度，可以实现自动特征选择：

$$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \text{diag}(\alpha_1^{-1}, ..., \alpha_d^{-1}))$$

其中 $\alpha_j$ 控制第 $j$ 个特征的相关性。通过最大化边际似然（证据）来学习这些超参数。

## 3.5 正则化的统一视角

### 3.5.1 MAP估计与正则化

从贝叶斯视角，正则化可以理解为MAP（最大后验）估计：

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_{\mathbf{w}} p(\mathbf{w}|\mathcal{D}) = \arg\max_{\mathbf{w}} [\log p(\mathcal{D}|\mathbf{w}) + \log p(\mathbf{w})]$$

不同的先验对应不同的正则化：
- 高斯先验 → $L_2$ 正则化（岭回归）
- 拉普拉斯先验 → $L_1$ 正则化（LASSO）
- 混合先验 → 弹性网络

### 3.5.2 偏差-方差分解视角

考虑期望预测误差：

$$\mathbb{E}[(y - \hat{y})^2] = \text{Bias}^2[\hat{y}] + \text{Var}[\hat{y}] + \sigma^2$$

正则化通过引入偏差来减少方差：
- 无正则化：低偏差，高方差
- 强正则化：高偏差，低方差
- 最优正则化：平衡偏差与方差

```
    误差
     ^
     |     总误差
     |    /     \
     |   /       \___
     |  /            \___  
     | /  方差           \___
     |/_____________________\___
     |      偏差²              
     |________________________
     0                      λ →
```

### 3.5.3 有效自由度

正则化减少了模型的有效自由度。对于岭回归：

$$\text{df}(\lambda) = \text{tr}(\mathbf{H}_\lambda) = \sum_{i=1}^d \frac{\sigma_i^2}{\sigma_i^2 + n\lambda}$$

其中 $\mathbf{H}_\lambda = \mathbf{X}(\mathbf{X}^T\mathbf{X} + n\lambda\mathbf{I})^{-1}\mathbf{X}^T$ 是帽子矩阵。

**Rule of thumb**：
- 有效自由度约为 $n/2$ 时通常泛化性能较好
- AIC准则：选择 $\lambda$ 使得 $\text{AIC} = n\log(\text{RSS}/n) + 2\text{df}(\lambda)$ 最小
- BIC准则：对大样本，使用 $\log(n)\text{df}(\lambda)$ 代替 $2\text{df}(\lambda)$

## 3.6 历史人物：高斯与最小二乘法

卡尔·弗里德里希·高斯（Carl Friedrich Gauss, 1777-1855）在1801年成功预测了谷神星的轨道，展示了最小二乘法的威力。

**谷神星轨道预测**：
- 1801年1月，皮亚齐发现谷神星，但40天后失踪
- 高斯仅用3个观测点，通过最小二乘法确定轨道
- 1801年12月，天文学家在高斯预测位置重新发现谷神星

**高斯的贡献**：
1. **正规方程**：将最小二乘转化为线性方程组求解
2. **误差理论**：证明了在高斯噪声假设下，最小二乘估计是最优无偏估计
3. **高斯消元法**：系统化的线性方程组求解方法

**高斯-马尔可夫定理**：在线性无偏估计类中，最小二乘估计具有最小方差。这为线性回归提供了坚实的理论基础。

## 3.7 现代连接：LoRA与大模型微调

### 3.7.1 低秩适应（LoRA）原理

LoRA（Low-Rank Adaptation）是一种参数高效的微调方法，其核心思想源于线性模型的低秩分解：

$$\mathbf{W}_{\text{new}} = \mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \mathbf{BA}$$

其中：
- $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$：预训练权重（冻结）
- $\mathbf{B} \in \mathbb{R}^{d \times r}$，$\mathbf{A} \in \mathbb{R}^{r \times k}$：低秩适应矩阵
- $r \ll \min(d, k)$：秩约束

### 3.7.2 与正则化的联系

LoRA可以视为一种结构化正则化：
- **参数约束**：限制更新在低维子空间
- **隐式正则化**：低秩约束防止过拟合
- **计算效率**：参数量从 $dk$ 降至 $(d+k)r$

```
    原始参数空间（高维）
         ／│＼
        ／ │ ＼
       ／  │  ＼
      ／   ↓   ＼
     └─ LoRA子空间 ─┘
        （低维）
```

### 3.7.3 实践中的经验法则

**Rule of thumb for LoRA**：
- 秩选择：$r = 4-64$，越大的模型可以用越小的秩
- 初始化：$\mathbf{A}$ 用高斯初始化，$\mathbf{B}$ 初始化为零
- 学习率：LoRA参数的学习率可以比基础模型高10-100倍
- 应用层：注意力层的 $Q, V$ 矩阵效果通常最好

**与传统方法的对比**：
| 方法 | 参数量 | 内存需求 | 适用场景 |
|------|--------|----------|----------|
| 全量微调 | $O(dk)$ | 高 | 数据充足 |
| 线性探测 | $O(k)$ | 低 | 特征已对齐 |
| LoRA | $O(r(d+k))$ | 中 | 数据有限 |
| 前缀微调 | $O(L \cdot d)$ | 中 | 序列任务 |

## 本章小结

**核心概念**：
1. **最小二乘法**：$\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
2. **岭回归**：$\hat{\mathbf{w}}_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
3. **LASSO软阈值**：$S_\lambda(x) = \text{sign}(x)\max(|x| - \lambda, 0)$
4. **贝叶斯后验**：正则化 = MAP估计with特定先验

**关键见解**：
- 正则化通过约束模型复杂度来改善泛化
- $L_1$ 正则化产生稀疏性，$L_2$ 正则化产生平滑性
- 贝叶斯方法提供不确定性量化
- 现代深度学习中的许多技术（如LoRA）本质上是正则化思想的延伸

**实用建议**：
- 特征数 > 样本数：使用弹性网络
- 需要特征选择：使用LASSO
- 需要稳定预测：使用岭回归
- 需要不确定性估计：使用贝叶斯方法

## 练习题

### 基础题

**3.1** 证明岭回归的解可以写成：
$$\hat{\mathbf{w}}_{\text{ridge}} = \sum_{i=1}^n \alpha_i \mathbf{x}_i$$
其中 $\boldsymbol{\alpha} = (\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y}$，$\mathbf{K}_{ij} = \mathbf{x}_i^T\mathbf{x}_j$ 是核矩阵。

<details>
<summary>提示</summary>
从正规方程出发，利用矩阵恒等式 $(\mathbf{A} + \lambda\mathbf{I})^{-1}\mathbf{A} = \mathbf{A}(\mathbf{A} + \lambda\mathbf{I})^{-1}$。
</details>

<details>
<summary>答案</summary>

从岭回归的正规方程：
$$\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

利用矩阵恒等式：
$$(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T = \mathbf{X}^T(\mathbf{X}\mathbf{X}^T + \lambda\mathbf{I})^{-1}$$

因此：
$$\hat{\mathbf{w}} = \mathbf{X}^T(\mathbf{X}\mathbf{X}^T + \lambda\mathbf{I})^{-1}\mathbf{y} = \mathbf{X}^T\boldsymbol{\alpha}$$

其中 $\boldsymbol{\alpha} = (\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y}$，这就是核岭回归的形式。
</details>

**3.2** 软阈值算子 $S_\lambda(x)$ 是如下优化问题的解：
$$\min_z \frac{1}{2}(z - x)^2 + \lambda|z|$$
请推导这个结果。

<details>
<summary>提示</summary>
分 $z > 0$, $z < 0$, $z = 0$ 三种情况讨论，利用次梯度条件。
</details>

<details>
<summary>答案</summary>

目标函数：$f(z) = \frac{1}{2}(z-x)^2 + \lambda|z|$

1. 当 $z > 0$：$f(z) = \frac{1}{2}(z-x)^2 + \lambda z$
   - 导数：$f'(z) = z - x + \lambda = 0$
   - 解：$z = x - \lambda$（需要 $x > \lambda$）

2. 当 $z < 0$：$f(z) = \frac{1}{2}(z-x)^2 - \lambda z$
   - 导数：$f'(z) = z - x - \lambda = 0$
   - 解：$z = x + \lambda$（需要 $x < -\lambda$）

3. 当 $z = 0$：需要 $0 \in \partial f(0) = [-x-\lambda, -x+\lambda]$
   - 即 $|x| \leq \lambda$

综合得：$S_\lambda(x) = \text{sign}(x)\max(|x| - \lambda, 0)$
</details>

**3.3** 对于正交设计矩阵 $\mathbf{X}^T\mathbf{X} = n\mathbf{I}$，比较最小二乘、岭回归和LASSO的解。

<details>
<summary>提示</summary>
在正交情况下，各坐标解耦，可以逐个求解。
</details>

<details>
<summary>答案</summary>

设 $\hat{\mathbf{w}}_{\text{OLS}} = \frac{1}{n}\mathbf{X}^T\mathbf{y}$ 为最小二乘解。

1. **岭回归**：
   $$\hat{w}_{j,\text{ridge}} = \frac{1}{1+\lambda}\hat{w}_{j,\text{OLS}}$$
   线性收缩所有系数

2. **LASSO**：
   $$\hat{w}_{j,\text{lasso}} = S_\lambda(\hat{w}_{j,\text{OLS}})$$
   软阈值处理，小系数变为0

3. **比较**：
   - OLS：不收缩
   - Ridge：比例收缩
   - LASSO：软阈值收缩+稀疏性
</details>

### 挑战题

**3.4** 考虑弹性网络的贝叶斯解释。什么样的先验分布对应于弹性网络正则化？这个先验是否共轭？

<details>
<summary>提示</summary>
考虑混合先验：高斯分布与拉普拉斯分布的结合。
</details>

<details>
<summary>答案</summary>

弹性网络对应的先验是高斯-拉普拉斯混合分布：

$$p(\mathbf{w}) \propto \exp\left(-\lambda\alpha\|\mathbf{w}\|_1 - \frac{\lambda(1-\alpha)}{2}\|\mathbf{w}\|_2^2\right)$$

这可以理解为：
$$p(w_j) \propto \exp(-\lambda\alpha|w_j|) \cdot \exp\left(-\frac{\lambda(1-\alpha)}{2}w_j^2\right)$$

这不是标准的共轭先验，因为拉普拉斯部分破坏了共轭性。因此需要使用近似推断方法（如变分推断或MCMC）来获得后验分布。

一种层次贝叶斯解释是引入潜变量 $\tau_j$：
- $w_j|\tau_j \sim \mathcal{N}(0, \tau_j)$
- $\tau_j$ 服从某个使得边际分布接近拉普拉斯的分布
</details>

**3.5** 设计一个自适应LASSO算法，其中每个特征的正则化参数 $\lambda_j$ 根据其重要性自动调整。

<details>
<summary>提示</summary>
考虑两阶段方法：先估计特征重要性，再据此调整惩罚。
</details>

<details>
<summary>答案</summary>

**自适应LASSO (Adaptive LASSO)**：

目标函数：
$$\mathcal{L}_{\text{adaptive}}(\mathbf{w}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \sum_{j=1}^d \lambda_j |w_j|$$

其中 $\lambda_j = \lambda / |\hat{w}_j^{\text{init}}|^\gamma$，$\gamma > 0$ 是调节参数。

**两阶段算法**：
1. **初始估计**：获得 $\hat{\mathbf{w}}^{\text{init}}$
   - 使用岭回归（稳定）
   - 或最小二乘（如果 $n > d$）

2. **自适应惩罚**：
   - 重要特征（大 $|\hat{w}_j^{\text{init}}|$）：小惩罚
   - 不重要特征（小 $|\hat{w}_j^{\text{init}}|$）：大惩罚

**理论性质**：
- Oracle性质：渐近等价于知道真实支撑集的估计
- 一致变量选择：正确识别零系数和非零系数

**实践建议**：
- $\gamma = 1$ 或 $2$ 通常效果良好
- 初始估计使用交叉验证选择的岭回归
- 对极小的 $|\hat{w}_j^{\text{init}}|$ 设置上限避免数值问题
</details>

**3.6** 推导并实现近端梯度下降算法求解LASSO问题，并分析其收敛速度。

<details>
<summary>提示</summary>
将LASSO分解为可微部分（平方损失）和不可微部分（$L_1$ 范数），使用近端算子。
</details>

<details>
<summary>答案</summary>

**近端梯度下降（Proximal Gradient Descent）**：

LASSO问题：$\min_{\mathbf{w}} f(\mathbf{w}) + g(\mathbf{w})$
- $f(\mathbf{w}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$（可微）
- $g(\mathbf{w}) = \lambda\|\mathbf{w}\|_1$（不可微）

**算法**：
```
初始化：w^(0) = 0, 步长 t
for k = 0, 1, 2, ... :
    # 梯度步
    z = w^(k) - t∇f(w^(k))
    # 近端步（软阈值）
    w^(k+1) = prox_{tg}(z) = S_{tλ}(z)
```

其中近端算子：
$$\text{prox}_{tg}(\mathbf{z}) = \arg\min_{\mathbf{w}} \left\{\frac{1}{2t}\|\mathbf{w} - \mathbf{z}\|_2^2 + g(\mathbf{w})\right\}$$

**收敛分析**：
- 步长选择：$t \leq 1/L$，其中 $L = \lambda_{\max}(\mathbf{X}^T\mathbf{X})/n$
- 收敛速度：$O(1/k)$
- 加速版本（FISTA）：$O(1/k^2)$

**FISTA（Fast Iterative Shrinkage-Thresholding Algorithm）**：
```
初始化：w^(0) = v^(0), θ^(0) = 1
for k = 0, 1, 2, ... :
    w^(k+1) = S_{tλ}(v^(k) - t∇f(v^(k)))
    θ^(k+1) = (1 + √(1 + 4(θ^(k))²))/2
    v^(k+1) = w^(k+1) + ((θ^(k)-1)/θ^(k+1))(w^(k+1) - w^(k))
```

**实践要点**：
- 使用回溯线搜索自适应选择步长
- 预计算 $\mathbf{X}^T\mathbf{X}$ 和 $\mathbf{X}^T\mathbf{y}$ 提高效率
- 设置停止准则：$\|\mathbf{w}^{(k+1)} - \mathbf{w}^{(k)}\|_\infty < \epsilon$
</details>

**3.7** 证明当 $n < d$ 时，LASSO最多选择 $n$ 个非零特征，而弹性网络无此限制。

<details>
<summary>提示</summary>
考虑KKT条件和解的维度。
</details>

<details>
<summary>答案</summary>

**LASSO的限制**：

KKT条件要求在最优解处：
$$\mathbf{X}^T(\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}) = n\lambda\mathbf{s}$$

其中 $\mathbf{s} \in \partial\|\hat{\mathbf{w}}\|_1$ 是次梯度，满足：
- $s_j = \text{sign}(\hat{w}_j)$ 若 $\hat{w}_j \neq 0$
- $s_j \in [-1, 1]$ 若 $\hat{w}_j = 0$

设 $\mathcal{A} = \{j: \hat{w}_j \neq 0\}$ 为活跃集，则：
$$\mathbf{X}_\mathcal{A}^T(\mathbf{y} - \mathbf{X}_\mathcal{A}\hat{\mathbf{w}}_\mathcal{A}) = n\lambda\mathbf{s}_\mathcal{A}$$

这要求 $\mathbf{X}_\mathcal{A}^T\mathbf{r} = n\lambda\mathbf{s}_\mathcal{A}$ 有解，其中 $\mathbf{r}$ 是残差。

由于 $\mathbf{X}_\mathcal{A}^T \in \mathbb{R}^{|\mathcal{A}| \times n}$，其秩最多为 $\min(|\mathcal{A}|, n)$。
当 $n < d$ 时，要使方程有唯一解，需要 $|\mathcal{A}| \leq n$。

**弹性网络的优势**：

弹性网络的KKT条件：
$$\mathbf{X}^T(\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}) = n\lambda(\alpha\mathbf{s} + (1-\alpha)\hat{\mathbf{w}})$$

$L_2$ 项的存在使得方程组总是有唯一解（通过正则化），不受 $n < d$ 的限制。

实际上，弹性网络等价于在增广数据集上的LASSO：
$$\tilde{\mathbf{X}} = \begin{bmatrix} \mathbf{X} \\ \sqrt{n\lambda(1-\alpha)}\mathbf{I} \end{bmatrix}, \quad \tilde{\mathbf{y}} = \begin{bmatrix} \mathbf{y} \\ \mathbf{0} \end{bmatrix}$$

增广后样本数变为 $n + d > d$，突破了LASSO的限制。
</details>

**3.8** 讨论正则化参数 $\lambda$ 的选择方法，包括交叉验证、AIC、BIC等，并比较它们的优缺点。

<details>
<summary>提示</summary>
考虑不同准则的理论基础和实践表现。
</details>

<details>
<summary>答案</summary>

**1. 交叉验证（CV）**：
- **方法**：将数据分成 $K$ 折，轮流作为验证集
- **选择**：$\lambda^* = \arg\min_\lambda \text{CV}(\lambda)$
- **优点**：直接优化预测性能，不需要分布假设
- **缺点**：计算密集，结果可能不稳定

**2. 信息准则**：

**AIC（赤池信息准则）**：
$$\text{AIC}(\lambda) = n\log(\text{RSS}(\lambda)/n) + 2\text{df}(\lambda)$$

**BIC（贝叶斯信息准则）**：
$$\text{BIC}(\lambda) = n\log(\text{RSS}(\lambda)/n) + \log(n)\text{df}(\lambda)$$

**3. 广义交叉验证（GCV）**：
$$\text{GCV}(\lambda) = \frac{\text{RSS}(\lambda)/n}{(1 - \text{df}(\lambda)/n)^2}$$

**比较**：

| 准则 | 理论基础 | 选择倾向 | 适用场景 |
|------|----------|----------|----------|
| CV | 预测误差最小化 | 适中 | 一般情况 |
| AIC | KL散度最小化 | 偏复杂 | 预测为主 |
| BIC | 后验概率最大化 | 偏简单 | 模型选择 |
| GCV | 留一交叉验证近似 | 类似CV | 线性模型 |

**实践建议**：
- 小样本（$n < 100$）：留一交叉验证
- 中等样本：5-10折交叉验证
- 大样本：BIC或2-3折交叉验证
- 高维稀疏：BIC倾向于选择正确的稀疏度

**一个标准误差规则**：
选择在最优 $\lambda^*$ 一个标准误差范围内的最大 $\lambda$：
$$\lambda_{\text{1se}} = \max\{\lambda: \text{CV}(\lambda) \leq \text{CV}(\lambda^*) + \text{SE}(\lambda^*)\}$$

这产生更稀疏、更稳定的模型。
</details>

## 常见陷阱与错误 (Gotchas)

1. **多重共线性被忽视**
   - **错误**：直接使用最小二乘，得到不稳定的巨大系数
   - **正确**：使用岭回归或弹性网络处理相关特征
   - **诊断**：计算VIF（方差膨胀因子），VIF > 10表示严重共线性

2. **标准化的重要性**
   - **错误**：不标准化特征直接使用正则化
   - **后果**：尺度大的特征受到更强惩罚
   - **正确**：正则化前标准化特征，预测时记得逆变换

3. **过度依赖默认参数**
   - **错误**：使用固定的 $\lambda = 1$
   - **正确**：通过交叉验证系统地选择
   - **技巧**：在对数尺度上搜索，如 $\lambda \in [10^{-4}, 10^2]$

4. **解释系数时忽略正则化影响**
   - **错误**：将LASSO系数解释为特征重要性
   - **事实**：正则化会缩小系数，不能直接比较大小
   - **正确**：使用置换重要性或其他方法评估特征重要性

5. **坐标下降的收敛假象**
   - **错误**：一轮迭代后就停止
   - **问题**：可能未收敛，特别是特征相关时
   - **正确**：检查相邻迭代的变化，设置合理的收敛阈值

6. **贝叶斯方法中的超参数选择**
   - **错误**：随意设置先验参数
   - **正确**：使用经验贝叶斯或完全贝叶斯方法
   - **实践**：通过边际似然最大化选择超参数

7. **高维情况下的过拟合**
   - **症状**：训练误差很小但测试误差很大
   - **解决**：增大正则化强度，使用稳定选择（Stability Selection）
   - **验证**：始终保留独立测试集

8. **数值稳定性问题**
   - **问题**：$\mathbf{X}^T\mathbf{X}$ 接近奇异
   - **解决**：使用QR分解或SVD代替正规方程
   - **预防**：添加小的岭正则化项 $\lambda = 10^{-6}$

---

返回[目录](index.md) | 下一章：[第4章：神经网络基础](chapter4.md)