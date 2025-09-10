# 第12章：深度强化学习

## 开篇段落

强化学习是人工智能中最接近"智能"本质的领域之一。不同于监督学习需要标注数据，强化学习智能体通过与环境交互，从试错中学习最优策略。当深度学习与强化学习结合，诞生了深度强化学习——这一技术不仅在游戏AI中取得惊人成就（从Atari游戏到围棋、星际争霸），更在机器人控制、自动驾驶、推荐系统等实际应用中展现巨大潜力。本章将从优化视角系统介绍深度强化学习的核心方法，重点关注价值函数近似、策略梯度、以及它们的结合——Actor-Critic架构。我们还将探讨AlphaGo的自我对弈机制、多智能体学习，以及RLHF在大语言模型对齐中的关键作用。

## 12.1 价值函数近似

### 12.1.1 从表格到函数近似

在传统强化学习中，我们用表格存储每个状态的价值函数：

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

但当状态空间巨大（如围棋的$10^{170}$种局面）或连续（如机器人关节角度）时，表格方法失效。解决方案是用函数近似器（如神经网络）表示价值函数：

$$Q(s,a;\theta) \approx Q^*(s,a)$$

这将强化学习问题转化为监督学习问题：最小化TD误差的平方：

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2]$$

其中$\theta^-$是目标网络参数，用于稳定训练。

### 12.1.2 深度Q网络（DQN）

DQN是深度强化学习的里程碑，首次在Atari游戏中达到人类水平。其核心创新包括：

**1. 经验回放（Experience Replay）**
```
经验池：D = {(s₁,a₁,r₁,s'₁), (s₂,a₂,r₂,s'₂), ..., (sₙ,aₙ,rₙ,s'ₙ)}
         ↓
   随机采样批次训练
```

打破样本相关性，提高样本效率，使训练更稳定。

**2. 目标网络（Target Network）**

使用独立的目标网络$Q(s,a;\theta^-)$计算TD目标，每隔C步更新：$\theta^- \leftarrow \theta$

这避免了"追逐移动目标"的问题，大幅提升训练稳定性。

**3. ε-贪婪探索**

以概率ε选择随机动作，以概率1-ε选择贪婪动作：

$$a = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s,a;\theta) & \text{with probability } 1-\epsilon
\end{cases}$$

典型设置：ε从1.0线性衰减到0.1，在100万步内完成。

### 12.1.3 DQN的改进

**Double DQN**：解决Q值过估计问题

标准DQN的目标：
$$y = r + \gamma \max_{a'} Q(s',a';\theta^-)$$

Double DQN的目标：
$$y = r + \gamma Q(s', \arg\max_{a'} Q(s',a';\theta);\theta^-)$$

动作选择用当前网络，价值评估用目标网络，有效减少过估计偏差。

**Dueling DQN**：分解价值函数

将Q函数分解为状态价值V和优势函数A：
$$Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + A(s,a;\theta,\alpha) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a';\theta,\alpha)$$

这种分解让网络更容易学习哪些状态重要，哪些动作带来额外优势。

**Rainbow DQN**：集成多种改进

Rainbow将七种改进集成：Double DQN、Dueling DQN、优先经验回放、多步学习、分布式RL、噪声网络、分类DQN。实践表明，优先经验回放和多步学习贡献最大。

## 12.2 策略梯度方法

### 12.2.1 REINFORCE算法

不同于价值方法学习Q函数再导出策略，策略梯度直接优化参数化策略$\pi_\theta(a|s)$：

目标函数（期望回报）：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \gamma^t r_t]$$

策略梯度定理给出梯度的解析形式：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) G_t]$$

其中$G_t = \sum_{k=t}^T \gamma^{k-t} r_k$是从时刻t开始的回报。

REINFORCE算法步骤：
1. 采样轨迹：$\tau = (s_0,a_0,r_0,...,s_T,a_T,r_T)$
2. 计算回报：$G_t = \sum_{k=t}^T \gamma^{k-t} r_k$
3. 更新参数：$\theta \leftarrow \theta + \alpha \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$

### 12.2.2 方差减少技术

REINFORCE的主要问题是方差过大，导致训练不稳定。关键改进：

**基线（Baseline）**：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t))]$$

常用基线是状态价值函数$V(s_t)$，这给出优势函数$A_t = G_t - V(s_t)$。

**重要性采样（Importance Sampling）**：

当用旧策略$\pi_{old}$的数据更新当前策略$\pi_\theta$时：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{old}}[\sum_{t=0}^T \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)} \nabla_\theta \log \pi_\theta(a_t|s_t) A_t]$$

### 12.2.3 近端策略优化（PPO）

PPO是目前最流行的策略梯度算法，在LLM的RLHF中广泛应用。核心思想是限制每次更新的步长：

**PPO-Clip目标函数**：
$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]$$

其中$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$是重要性采样比率。

```
         r_t(θ)
           │
    ┌──────┼──────┐
    │      │      │
  1-ε      1     1+ε
    └──────┴──────┘
       clip区间
```

当优势为正（A > 0）时，鼓励增大该动作概率，但限制在1+ε内；
当优势为负（A < 0）时，鼓励减小该动作概率，但限制在1-ε内。

典型超参数：ε = 0.2，即每次更新策略变化不超过20%。

## 12.3 Actor-Critic架构

### 12.3.1 基本架构

Actor-Critic结合了价值方法和策略方法的优点：
- Actor（演员）：策略网络$\pi_\theta(a|s)$，负责选择动作
- Critic（评论家）：价值网络$V_\phi(s)$或$Q_\psi(s,a)$，负责评估动作

```
状态 s ──┬──→ Actor π(a|s) ──→ 动作 a
         │                         ↓
         └──→ Critic V(s) ──→ 价值估计
                ↑                  │
                └──────────────────┘
                   TD误差用于更新
```

更新规则：
- Critic更新：最小化TD误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
- Actor更新：$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \delta_t$

### 12.3.2 异步优势Actor-Critic（A3C）

A3C通过并行化提升训练效率：

```
    Global Network
    θ_global, φ_global
           ↑
    异步更新梯度
     ↗  ↗  ↗  ↗
Worker1 Worker2 ... WorkerN
  ↓       ↓           ↓
 Env1    Env2   ...  EnvN
```

每个worker独立与环境交互，定期将梯度推送到全局网络。这种异步更新自然引入探索多样性，无需经验回放。

### 12.3.3 软Actor-Critic（SAC）

SAC引入最大熵框架，在优化期望回报的同时最大化策略熵：

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}[\sum_{t=0}^T (r_t + \alpha \mathcal{H}(\pi(\cdot|s_t)))]$$

其中$\mathcal{H}(\pi(\cdot|s_t)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s_t)]$是策略熵。

优点：
1. 鼓励探索（高熵意味着更随机的动作选择）
2. 提高鲁棒性（避免过早收敛到次优策略）
3. 自动调节探索-利用平衡

温度参数α可自适应调节：
$$\alpha_t = \alpha_{t-1} - \lambda \nabla_\alpha (\mathcal{H}(\pi) - \mathcal{H}_{target})$$

使策略熵维持在目标水平$\mathcal{H}_{target}$附近。

## 12.4 AlphaGo与自我对弈

### 12.4.1 蒙特卡洛树搜索（MCTS）

AlphaGo的核心是将深度学习与MCTS结合。MCTS通过模拟未来对局评估当前局面：

```
        根节点(当前局面)
         /    |    \
       /      |      \
     动作1  动作2   动作3
      /\      |       /\
    ...     叶节点   ...
            
MCTS四步骤：
1. 选择(Selection)：从根节点向下选择
2. 扩展(Expansion)：添加新节点
3. 模拟(Simulation)：随机走子到终局
4. 回传(Backpropagation)：更新路径上所有节点
```

节点选择使用UCT（Upper Confidence Bound for Trees）：
$$a = \arg\max_a \left( Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}} \right)$$

第一项是利用（选择高价值动作），第二项是探索（选择访问次数少的动作）。

### 12.4.2 AlphaGo的神经网络

AlphaGo使用两个网络指导MCTS：

**策略网络$p_\sigma(a|s)$**：预测人类专家的落子
- 输入：19×19×48的特征（棋盘状态、气、打吃等）
- 输出：361个位置的概率分布
- 训练：先用3000万人类棋谱监督学习，再通过自我对弈强化学习

**价值网络$v_\theta(s)$**：评估局面胜率
- 输入：与策略网络相同
- 输出：当前玩家的胜率[-1, 1]
- 训练：用自我对弈产生的局面和最终结果

### 12.4.3 AlphaGo Zero：纯自我对弈

AlphaGo Zero完全摒弃人类知识，从随机初始化开始：

**统一网络架构**：
$$(p, v) = f_\theta(s)$$

同时输出策略和价值，共享特征提取层。

**自我对弈训练循环**：
1. 用当前网络+MCTS进行自我对弈，生成训练数据
2. 训练网络最小化损失：
   $$\mathcal{L} = (z - v)^2 - \pi^T \log p + c||\theta||^2$$
   其中z是实际游戏结果，π是MCTS改进的策略
3. 评估新网络，如果胜率>55%则更新

**关键创新**：
- 残差网络架构（相比AlphaGo的普通卷积）
- MCTS中用网络预测代替随机rollout
- 数据增强：利用围棋的8重对称性

仅用4个TPU训练3天，AlphaGo Zero就超越了AlphaGo Master。

### 12.4.4 MuZero：无模型的规划

MuZero将AlphaGo的方法推广到未知规则的环境：

三个网络：
- 表示网络：$h = h_\theta(o_1,...,o_t)$ 将观察历史编码为隐状态
- 动态网络：$h', r = g_\theta(h, a)$ 预测下一隐状态和奖励
- 预测网络：$(p, v) = f_\theta(h)$ 输出策略和价值

在隐空间中进行MCTS规划，无需知道真实环境动态。

## 12.5 OpenAI Five与多智能体学习

### 12.5.1 Dota 2的挑战

Dota 2比围棋复杂得多：
- 状态空间：~20,000维连续观察
- 动作空间：170,000种离散动作
- 部分可观察：战争迷雾
- 长期信用分配：平均45分钟，~20,000个决策
- 团队协作：5v5需要复杂配合

### 12.5.2 大规模分布式训练

OpenAI Five的训练规模前所未有：

```
     Optimizer (CPU)
          ↓
    参数更新 (Adam)
          ↓
   ┌──────────────┐
   │  GPU Workers │ × 1024
   │   前向传播   │
   └──────────────┘
          ↓
   ┌──────────────┐
   │ CPU Rollouts │ × 128,000
   │   环境交互   │
   └──────────────┘

每天相当于180年游戏时间
```

使用PPO算法，但有关键修改：
- 大批量：批大小60,000样本
- 长时间范围：16,000步的GAE
- 快速迭代：每2分钟更新一次模型

### 12.5.3 涌现的团队策略

训练过程中观察到策略演化：
1. **初期**（~2周）：学会基本操作，随机游走
2. **中期**（~1月）：出现简单配合，如2v1 gank
3. **后期**（~2月）：复杂团战配合，假动作欺骗

令人惊讶的涌现行为：
- **牺牲**：辅助主动送死让核心英雄获得经验
- **诱饵**：故意暴露引诱敌人进入埋伏
- **经济分配**：自动形成1号位到5号位的资源分配

### 12.5.4 多智能体学习的关键技术

**集中训练，分散执行**：
- 训练时共享全局信息优化团队奖励
- 执行时每个智能体独立决策

**团队奖励塑形**：
```python
team_reward = 游戏胜负 + 0.5×团队击杀 + 0.3×推塔 + 0.2×经济领先
individual_reward = 0.7×team_reward + 0.3×individual_contribution
```

**课程学习**：
- 开始：限制英雄池（5个英雄）
- 逐步增加：物品系统、技能升级、更多英雄
- 最终：完整游戏规则

## 12.6 历史人物：理查德·萨顿与强化学习的现代框架

理查德·萨顿（Richard Sutton）被誉为"强化学习之父"。他的贡献奠定了现代强化学习的理论基础。

**时序差分学习（TD Learning）**：
萨顿在1988年提出TD(λ)算法，统一了蒙特卡洛方法和动态规划：
$$V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$$

这个简洁的更新规则成为几乎所有现代RL算法的基础。

**《强化学习导论》**：
与Andrew Barto合著的教科书成为领域圣经，定义了标准术语和数学框架：
- 马尔可夫决策过程（MDP）
- 价值函数与贝尔曼方程
- 探索与利用的权衡

**"苦涩的教训"（The Bitter Lesson, 2019）**：
萨顿总结了AI研究70年的经验教训：

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective."

核心观点：
1. 通用方法+大规模计算 > 人类知识编码
2. 搜索和学习是最重要的两个通用方法
3. 摩尔定律使计算指数增长，应该顺势而为

这篇文章深刻影响了大模型时代的研究方向。

**对深度RL的预见**：
早在2015年DQN发表前，萨顿就预言函数近似将revolutionize RL。他提出的"预测+控制"框架完美适配深度学习时代。

## 12.7 现代连接：RLHF在LLM对齐中的应用

### 12.7.1 从预训练到对齐

大语言模型的训练分为两个阶段：

**预训练**：最大似然估计
$$\mathcal{L}_{MLE} = -\mathbb{E}_{x \sim \mathcal{D}}[\log p_\theta(x)]$$

预训练模型学会了语言建模，但可能生成有害、偏见或低质量内容。

**对齐**：使模型行为符合人类价值观
- Helpful（有用）：准确回答问题，完成任务
- Harmless（无害）：拒绝有害请求，避免偏见
- Honest（诚实）：承认不确定性，不编造信息

### 12.7.2 RLHF的三步流程

**Step 1: 监督微调（SFT）**
收集高质量的(prompt, response)对，微调预训练模型：
$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{demo}}[\log \pi_{SFT}(y|x)]$$

典型数据规模：10K-100K高质量样本。

**Step 2: 奖励模型训练**
收集人类偏好数据：对同一prompt的多个response进行排序。

训练奖励模型预测人类偏好：
$$\mathcal{L}_R = -\mathbb{E}_{(x,y_w,y_l)}[\log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))]$$

其中$y_w$是偏好的回复，$y_l$是较差的回复。

**Step 3: PPO优化**
用强化学习优化语言模型，最大化奖励同时避免偏离SFT模型：
$$\mathcal{L}_{PPO} = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}[r_\phi(x,y)] - \beta \cdot KL[\pi_\theta || \pi_{SFT}]$$

KL惩罚项防止模型"过度优化"奖励函数（reward hacking）。

### 12.7.3 RLHF的关键技术细节

**PPO在LLM中的适配**：

将生成视为序列决策问题：
- 状态$s_t$：已生成的token序列
- 动作$a_t$：下一个token
- 奖励：仅在序列结束时给出（稀疏奖励）

```
Prompt: "写一首诗"
    ↓
State₀: [BOS]
Action₀: "春"
    ↓
State₁: [BOS, 春]
Action₁: "风"
    ↓
    ...
    ↓
StateT: [BOS, 春, 风, ..., 。]
Reward: r(prompt, full_response)
```

**价值函数初始化**：
用SFT模型初始化Actor，用奖励模型初始化Critic的最后一层，加速收敛。

**优势估计**：
由于奖励稀疏，使用奖励模型的中间激活作为伪奖励：
$$r_t = \begin{cases}
r_\phi(x, y_{1:T}) & t = T \\
\gamma V(s_{t+1}) - V(s_t) & t < T
\end{cases}$$

### 12.7.4 Constitutional AI：自我改进的RLHF

Anthropic提出的Constitutional AI让模型自我批评和改进：

**第一阶段：Constitutional SFT**
1. 生成初始回复
2. 让模型自我批评："这个回复是否包含有害内容？"
3. 如果有问题，让模型修正
4. 用修正后的回复进行SFT

**第二阶段：Constitutional RLHF**
用AI反馈替代人类反馈：
1. 生成多个回复
2. 让模型根据宪法原则评分
3. 训练奖励模型
4. PPO优化

宪法原则示例：
- "选择最有帮助、无害、诚实的回复"
- "避免给出可能造成伤害的建议"
- "如果不确定，承认局限性而非编造"

### 12.7.5 RLHF的挑战与未来

**奖励模型的局限**：
- 分布外泛化差：训练时未见过的prompt类型
- 奖励操纵：模型学会欺骗奖励函数
- 人类反馈噪声：标注者之间分歧大

**直接偏好优化（DPO）**：
绕过奖励模型，直接从偏好数据优化策略：
$$\mathcal{L}_{DPO} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$$

DPO将RLHF简化为单一的监督学习问题，训练更稳定。

**在线RLHF**：
实时收集用户反馈，持续改进：
```
用户交互 → 隐式/显式反馈 → 更新奖励模型 → PPO微调
     ↑                                      ↓
     └──────────── 部署新模型 ←─────────────┘
```

## 12.8 本章小结

深度强化学习将深度学习的表征能力与强化学习的决策框架结合，在游戏AI、机器人控制、对话系统等领域取得突破性进展。

**核心概念回顾**：

1. **价值函数近似**：用神经网络表示Q函数或V函数，DQN通过经验回放和目标网络实现稳定训练

2. **策略梯度**：直接优化参数化策略，REINFORCE提供基础算法，PPO通过限制更新步长提高稳定性

3. **Actor-Critic**：结合价值和策略方法的优点，A3C实现高效并行训练，SAC引入最大熵框架

4. **自我对弈**：AlphaGo展示了MCTS与深度学习的强大结合，通过自我对弈达到超人水平

5. **多智能体学习**：OpenAI Five展示了简单RL算法+大规模计算的威力，涌现复杂团队行为

6. **RLHF**：将RL应用于LLM对齐，通过人类反馈优化模型行为，成为ChatGPT成功的关键

**关键公式汇总**：

- TD误差：$\delta = r + \gamma V(s') - V(s)$
- DQN损失：$\mathcal{L} = (r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2$
- 策略梯度：$\nabla J = \mathbb{E}[\nabla \log \pi(a|s) A(s,a)]$
- PPO目标：$L = \min(r(\theta)A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A)$
- RLHF目标：$J = \mathbb{E}[r(x,y)] - \beta \cdot KL[\pi||\pi_{ref}]$

## 12.9 常见陷阱与错误

### 陷阱1：奖励工程不当

**问题**：设计的奖励函数被智能体"hack"
```
例：训练机器人抓取物体
错误奖励：r = 手臂高度
结果：机器人学会举手而非抓取
```

**解决方案**：
- 稀疏奖励：仅在完成任务时给奖励
- 奖励塑形要谨慎：确保不改变最优策略
- 逆强化学习：从专家演示中学习奖励函数

### 陷阱2：探索不足

**问题**：智能体陷入局部最优，不愿尝试新动作

**解决方案**：
- ε-贪婪探索：保持最小探索率（如0.01）
- 熵正则化：如SAC中的最大熵框架
- 好奇心驱动：内在奖励鼓励探索新状态
- 噪声注入：在动作或参数中加入噪声

### 陷阱3：分布偏移

**问题**：训练分布与测试分布不一致
```
训练：从固定起点开始
测试：从任意状态开始
结果：智能体在新起点失败
```

**解决方案**：
- 域随机化：训练时随机化环境参数
- 课程学习：逐步增加任务难度
- 在线学习：部署后继续学习适应

### 陷阱4：梯度估计方差过大

**问题**：REINFORCE等算法方差极大，训练不稳定

**诊断方法**：
```python
# 监控梯度范数
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
if grad_norm > 10:
    print("Warning: Large gradient detected")
```

**解决方案**：
- 使用基线减少方差
- 增大批量大小
- 梯度裁剪
- 使用GAE（Generalized Advantage Estimation）

### 陷阱5：超参数敏感性

**问题**：RL算法对超参数极其敏感

**关键超参数及经验值**：
- 学习率：3e-4（PPO）、1e-3（DQN）
- 折扣因子γ：0.99（长期任务）、0.95（短期任务）
- PPO clip范围：0.1-0.3
- 目标网络更新频率：1000-10000步

**调试技巧**：
1. 先在简单环境（CartPole）验证实现
2. 使用网格搜索或贝叶斯优化
3. 记录所有超参数用于复现

### 陷阱6：样本效率低

**问题**：需要海量交互才能学会简单任务

**解决方案**：
- 使用off-policy算法（如SAC）重用旧数据
- 模型based RL：学习环境模型减少真实交互
- 迁移学习：从相关任务预训练
- 演示数据：结合模仿学习

### 陷阱7：训练不稳定

**症状**：
- 性能突然崩溃
- 学习曲线剧烈震荡
- Q值发散到无穷

**诊断检查清单**：
1. 检查奖励是否有界
2. 监控KL散度（PPO）
3. 检查目标网络是否正常更新
4. 验证经验回放是否正确采样
5. 确认梯度没有爆炸或消失

## 12.10 练习题

### 基础题

**练习12.1** TD误差与蒙特卡洛误差
给定轨迹：$s_0 \xrightarrow{a_0, r_0=1} s_1 \xrightarrow{a_1, r_1=2} s_2 \xrightarrow{a_2, r_2=3} s_{终止}$，
当前价值估计：$V(s_0)=2, V(s_1)=3, V(s_2)=1$，折扣因子$\gamma=0.9$。

计算：
a) $s_0$的TD误差
b) $s_0$的蒙特卡洛误差
c) 解释两者的区别

<details>
<summary>提示</summary>
TD误差使用下一状态的估计值，蒙特卡洛使用实际回报。
</details>

<details>
<summary>答案</summary>

a) TD误差：$\delta_0 = r_0 + \gamma V(s_1) - V(s_0) = 1 + 0.9 \times 3 - 2 = 1.7$

b) 蒙特卡洛回报：$G_0 = 1 + 0.9 \times 2 + 0.81 \times 3 = 5.23$
   蒙特卡洛误差：$G_0 - V(s_0) = 5.23 - 2 = 3.23$

c) TD误差只使用一步信息，偏差大但方差小；蒙特卡洛使用完整轨迹，无偏但方差大。
</details>

**练习12.2** DQN的目标网络
解释为什么DQN需要目标网络。如果直接用$y = r + \gamma \max_{a'} Q(s',a';\theta)$作为目标会发生什么？

<details>
<summary>提示</summary>
考虑目标依赖于正在优化的参数时会发生什么。
</details>

<details>
<summary>答案</summary>
不使用目标网络会导致：
1. 目标值随每次更新而变化，像"追逐移动的目标"
2. 高估偏差加剧：网络倾向于高估Q值，更新又基于这些高估值
3. 训练不稳定甚至发散：正反馈循环导致Q值爆炸

目标网络通过固定目标一段时间，打破这种循环，提供稳定的优化目标。
</details>

**练习12.3** 策略梯度的推导
证明策略梯度定理：$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) G_t]$

<details>
<summary>提示</summary>
使用对数导数技巧：$\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$
</details>

<details>
<summary>答案</summary>
目标函数：$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int p_\theta(\tau) R(\tau) d\tau$

求梯度：
$$\nabla_\theta J = \nabla_\theta \int p_\theta(\tau) R(\tau) d\tau = \int \nabla_\theta p_\theta(\tau) R(\tau) d\tau$$

使用对数导数技巧：
$$= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) d\tau = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log p_\theta(\tau) R(\tau)]$$

其中$p_\theta(\tau) = p(s_0) \prod_t \pi_\theta(a_t|s_t) p(s_{t+1}|s_t,a_t)$

因此：$\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$

最终得到策略梯度定理。
</details>

### 挑战题

**练习12.4** PPO的clip机制分析
PPO的clip目标函数为：$L = \min(r(\theta)A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A)$

分析以下四种情况下的梯度：
a) $A > 0, r(\theta) > 1+\epsilon$
b) $A > 0, r(\theta) < 1-\epsilon$  
c) $A < 0, r(\theta) > 1+\epsilon$
d) $A < 0, r(\theta) < 1-\epsilon$

<details>
<summary>提示</summary>
画出目标函数关于$r(\theta)$的图像，分析在不同区间的行为。
</details>

<details>
<summary>答案</summary>

a) $A > 0, r(\theta) > 1+\epsilon$：
   目标被clip在$(1+\epsilon)A$，梯度为0，防止过度增大动作概率

b) $A > 0, r(\theta) < 1-\epsilon$：
   取$\min$后为$r(\theta)A$，有正梯度，鼓励增大动作概率

c) $A < 0, r(\theta) > 1+\epsilon$：
   取$\min$后为$r(\theta)A$（负值），有负梯度，鼓励减小动作概率

d) $A < 0, r(\theta) < 1-\epsilon$：
   目标被clip在$(1-\epsilon)A$，梯度为0，防止过度减小动作概率

总结：PPO允许向好的方向改进，但限制每次更新的幅度。
</details>

**练习12.5** RLHF中的分布偏移
在RLHF训练中，策略$\pi_\theta$会逐渐偏离初始策略$\pi_{SFT}$。如果没有KL惩罚项，会出现什么问题？设计实验验证你的假设。

<details>
<summary>提示</summary>
考虑奖励模型的训练分布和泛化能力。
</details>

<details>
<summary>答案</summary>

没有KL惩罚会导致"奖励黑客"问题：

1. **分布偏移**：$\pi_\theta$生成的文本越来越不像训练奖励模型时的分布
2. **奖励模型失效**：在分布外数据上，奖励模型给出不可靠的高分
3. **模式坍塌**：模型学会生成特定模式以最大化（错误的）奖励

实验设计：
```python
# 训练两个模型：有/无KL惩罚
model_with_kl = train_rlhf(kl_coeff=0.01)
model_without_kl = train_rlhf(kl_coeff=0.0)

# 评估指标
# 1. 困惑度：测量偏离原始分布程度
# 2. 多样性：测量生成文本的重复性
# 3. 人类评估：真实质量 vs 奖励模型分数

# 预期结果：
# model_without_kl会有更高的奖励分数但更低的真实质量
```
</details>

**练习12.6** AlphaGo的MCTS改进
标准MCTS使用随机rollout估计叶节点价值。AlphaGo Zero用价值网络$v_\theta(s)$替代。分析这种改进的优缺点，并提出一种结合两者优点的混合方法。

<details>
<summary>提示</summary>
考虑计算效率、估计偏差和方差。
</details>

<details>
<summary>答案</summary>

**随机rollout**：
- 优点：无偏估计（给定足够模拟）
- 缺点：方差大，计算昂贵，需要大量模拟

**价值网络**：
- 优点：快速评估，方差小
- 缺点：有偏差，泛化到新局面可能不准

**混合方法**：
$$v_{mix}(s) = \lambda \cdot v_\theta(s) + (1-\lambda) \cdot v_{rollout}(s)$$

自适应权重：
- 开局和中局：更依赖价值网络（$\lambda = 0.8$）
- 终局：更依赖rollout（$\lambda = 0.3$）
- 不确定性高时：增加rollout权重

```python
def evaluate_leaf(state, model, uncertainty_threshold=0.5):
    value_nn = model.predict_value(state)
    uncertainty = model.predict_uncertainty(state)
    
    if uncertainty > uncertainty_threshold:
        # 高不确定性，使用rollout
        value_rollout = monte_carlo_rollout(state, n_simulations=10)
        return 0.3 * value_nn + 0.7 * value_rollout
    else:
        # 低不确定性，信任网络
        return value_nn
```
</details>

**练习12.7** 多智能体学习的纳什均衡
考虑两个智能体的零和游戏。证明如果两个智能体都使用策略梯度且学习率足够小，它们会收敛到纳什均衡。在什么条件下这个结论不成立？

<details>
<summary>提示</summary>
考虑梯度动力学和不动点。
</details>

<details>
<summary>答案</summary>

**收敛到纳什均衡的条件**：

设两个智能体策略为$\pi_1, \pi_2$，零和游戏中：
$$J_1(\pi_1, \pi_2) = -J_2(\pi_1, \pi_2)$$

策略梯度更新：
$$\pi_1^{t+1} = \pi_1^t + \alpha \nabla_{\pi_1} J_1(\pi_1^t, \pi_2^t)$$
$$\pi_2^{t+1} = \pi_2^t + \alpha \nabla_{\pi_2} J_2(\pi_1^t, \pi_2^t)$$

在纳什均衡$(\pi_1^*, \pi_2^*)$处，梯度为零：
$$\nabla_{\pi_1} J_1(\pi_1^*, \pi_2^*) = 0, \quad \nabla_{\pi_2} J_2(\pi_1^*, \pi_2^*) = 0$$

**收敛性分析**：
- 学习率足够小时，可用ODE近似离散更新
- 零和游戏的梯度场具有保守性质
- Lyapunov函数存在，保证收敛

**不成立的条件**：
1. **非零和游戏**：可能有多个均衡，循环动力学
2. **函数近似**：神经网络引入近似误差
3. **异步更新**：破坏梯度场的保守性
4. **探索噪声**：持续的探索阻止精确收敛

实践中常见的循环现象（如石头剪刀布）说明了这些理论限制。
</details>

**练习12.8** 设计一个RL系统
设计一个强化学习系统来优化大语言模型的推理能力。要求：
1. 定义状态、动作、奖励
2. 选择合适的RL算法并说明理由
3. 描述训练流程
4. 讨论潜在挑战和解决方案

<details>
<summary>提示</summary>
参考RLHF的成功经验，但针对推理任务的特点进行调整。
</details>

<details>
<summary>答案</summary>

**系统设计：思维链强化学习（Chain-of-Thought RL）**

**1. MDP定义**：
- 状态：$s_t = $ (问题, 已生成的推理步骤)
- 动作：$a_t = $ 下一个推理步骤或最终答案
- 奖励：
  ```
  r(s,a) = {
    +1  如果最终答案正确
    -0.01 * len(step)  长度惩罚
    +0.1  如果推理步骤被验证器确认有效
  }
  ```

**2. 算法选择：PPO + 专家迭代**
- PPO：稳定性好，适合大规模语言模型
- 专家迭代：用MCTS搜索更好的推理路径

**3. 训练流程**：

```python
# Phase 1: 监督学习
model = finetune_on_reasoning_datasets(base_model)

# Phase 2: 自我对弈生成数据
for problem in training_problems:
    # 使用MCTS搜索最佳推理路径
    reasoning_paths = mcts_search(model, problem, n_simulations=100)
    best_path = select_correct_path(reasoning_paths)
    dataset.add(problem, best_path)

# Phase 3: PPO微调
for epoch in range(n_epochs):
    # 采样问题
    problems = sample_problems()
    
    # 生成推理
    trajectories = model.generate_reasoning(problems)
    
    # 计算奖励（使用验证器或ground truth）
    rewards = compute_rewards(trajectories)
    
    # PPO更新
    ppo_update(model, trajectories, rewards)
```

**4. 挑战与解决方案**：

挑战1：推理步骤难以自动评估
- 解决：训练独立的步骤验证器
- 使用形式化验证（数学问题）
- 人在回路的主动学习

挑战2：探索空间巨大
- 解决：课程学习，从简单到复杂
- 使用提示工程引导探索
- 保持推理多样性的熵正则化

挑战3：奖励稀疏（只有最终答案有奖励）
- 解决：中间奖励塑形
- 使用思维链蒸馏的伪奖励
- 价值函数初始化用SFT模型

挑战4：灾难性遗忘
- 解决：保持在原始模型的KL球内
- 定期在一般任务上评估
- 混合训练数据

**评估指标**：
- 准确率提升
- 推理步骤的可解释性
- 泛化到新问题类型的能力
- 推理效率（步骤数）
</details>
