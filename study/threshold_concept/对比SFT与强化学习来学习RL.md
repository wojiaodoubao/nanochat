# 对比 SFT 与强化学习来学习 RL

本文从一个统一问题出发理解大模型训练：**模型怎样从一批样本中得到一个可优化的 loss，并通过反向传播改变参数？**

预训练、SFT 和强化学习最终都会执行类似的底层步骤：

```python
loss.backward()
optimizer.step()
```

真正不同的是：

1. 训练样本从哪里来；
2. 怎样判断模型输出是好还是坏；
3. 如何把这种好坏转成每个 token 对参数梯度的贡献。

本文刻意沿用我们讨论时的思考顺序：先理解 SFT 中“每个 token 都有标准答案”，再理解 RL 中“只有整个回答的结果奖励”，最后看 PPO、GRPO 如何把奖励变成可以反向传播的 loss。

---

## 1. 先校正三个容易混淆的术语

### 1.1 预训练通常称为自监督学习，而不是传统人工标注的有监督学习

大语言模型预训练使用普通文本自动构造监督信号。例如：

```text
原始文本：我 喜欢 猫
输入：    我 喜欢
目标：       喜欢 猫
```

目标 token 直接来自文本自身，不需要人工逐个标注，所以通常称为 **self-supervised learning（自监督学习）**。但从 loss 计算角度看，它确实和普通监督学习一样：每个被训练的位置都有一个确定的 target token。

SFT 则是更典型的监督学习：给定用户问题和人工、Teacher 或数据集提供的标准 Assistant 回答，只在 Assistant 应生成的位置计算监督 loss。

因此，下文为了突出 loss 结构，有时会把预训练和 SFT 合称为“有明确 token target 的训练”，但二者的数据来源仍需区分：

- 预训练：文本自身产生标签，属于自监督学习；
- SFT：外部提供目标回答，属于监督微调。

### 1.2 RL 不是无监督学习

强化学习虽然没有逐 token 标准答案，但仍然有监督信号，只是信号形式从 `target token` 变成了 `reward`、环境反馈或偏好评分。

所以更准确的区别是：

```text
预训练 / SFT：每个训练 token 有明确 target
RL：没有唯一的 target 序列，只有行为结果的 reward
```

### 1.3 Critic 与组均值 baseline 不是一回事

PPO 常配合 **Critic（价值网络）**估计状态价值，用它计算 advantage。

GRPO 则正是为了不训练 Critic：同一个问题生成一组回答，用这一组 reward 的均值和标准差构造相对 advantage。

nanochat 更简单：既没有 Critic，也没有除以标准差，只使用：

\[
A_i=r_i-\bar r
\]

因此不能把“组均值方法”叫作 Critic 网络。它们的用途相似——都为 policy gradient 提供 baseline、降低方差——但实现不同。

---

## 2. 强化学习与 SFT 的本质区别

## 2.1 相同点：最终都要构造 loss 并做梯度更新

无论 SFT 还是 RL，模型参数都不能被“正确”“错误”这样的自然语言直接修改。训练程序必须先构造一个可微的标量目标：

```text
样本 / rollout
    ↓
计算 loss
    ↓
loss.backward()
    ↓
得到每个参数的 gradient
    ↓
optimizer.step()
    ↓
模型参数变化
```

两者都使用相同的神经网络、自动微分和优化器基础设施。

## 2.2 SFT：学习“这个位置应该输出什么”

SFT 数据里存在标准回答。假设标准回答为：

```text
我 喜欢 猫
```

模型依次学习：

```text
看到“我”          → 下一个 token 应该是“喜欢”
看到“我 喜欢”     → 下一个 token 应该是“猫”
```

第 \(t\) 个位置有明确 target \(y_t\)，模型输出整个词表的概率分布，训练只取标准 token 的概率：

\[
L_t=-\log p_\theta(y_t\mid x_{<t})
\]

如果标准答案 token 的概率越大，loss 越小：

| 标准 token 概率 | token loss |
|---:|---:|
| 0.9 | \(-\log 0.9\approx0.105\) |
| 0.5 | \(-\log 0.5\approx0.693\) |
| 0.1 | \(-\log 0.1\approx2.303\) |
| 接近 0 | loss 非常大 |

多个有效 token 的 SFT loss 是平均交叉熵：

\[
L_{\mathrm{SFT}}
=-\frac{1}{N}\sum_{t:m_t=1}\log p_\theta(y_t\mid x_{<t})
\]

其中 \(m_t\) 是 loss mask。在 nanochat 中，用户问题、外部工具返回和 padding 不参与训练，target 会被设置为 `-1`；只有 Assistant 自己应生成的 token 参与 loss。更多实现细节见 [`../stf/sft.md`](../stf/sft.md)。

## 2.3 RL：学习“哪些生成轨迹更值得提高概率”

RL 没有唯一标准回答。例如一道数学题可能存在多种正确推理过程：

```text
回答 A：先列方程，再求解，最终答案正确
回答 B：换一种推导，最终答案也正确
回答 C：推导看似合理，但最终答案错误
```

训练系统不要求模型逐字模仿 A 或 B，而是给整条回答一个 reward：

```text
答对：reward = 1
答错：reward = 0
```

Policy Gradient 的基本目标可以写成：

\[
L_{\mathrm{PG}}
=-\frac{1}{N}\sum_t A_t\log p_\theta(a_t\mid s_t)
\]

其中：

- \(s_t\)：当前状态，对 LLM 来说是 prompt 加已经生成的 token；
- \(a_t\)：模型在第 \(t\) 步实际采样出来的 token；
- \(p_\theta(a_t\mid s_t)\)：模型生成这个 token 的概率；
- \(A_t\)：advantage，表示这次行为相对 baseline 好多少。

如果 \(A_t>0\)，梯度更新会提高实际采样 token 的概率；如果 \(A_t<0\)，会降低其概率。

### RL 的核心困难：Credit Assignment

最终答案正确，只能说明整个生成轨迹总体成功，无法直接知道：

- 哪个 token 是关键转折；
- 哪个推理步骤真正带来了正确答案；
- 哪些 token 只是无关措辞；
- 错误答案究竟在哪一步开始偏离。

所以 RL 要解决的核心问题之一，就是如何把序列级或环境级 reward 转成 token/动作级 advantage。

nanochat 采用最简单的方案：同一回答中的所有生成 token 共用同一个序列 advantage。PPO 则常用 Critic 和 GAE 给不同时间步估计更细粒度的 advantage；GRPO 用同题多回答之间的相对 reward 代替 Critic。

---

## 3. Cross Entropy：SFT 的 loss 如何使用 token 概率

语言模型在每个位置输出的不是一个 token，而是一个形状为 `vocab_size` 的 logits 向量。经过 softmax 后得到完整概率分布：

```text
P(猫)   = 0.60
P(狗)   = 0.30
P(汽车) = 0.10
```

假设标准答案是“猫”，交叉熵只显式取：

\[
L=-\log P(猫)
\]

但其他 token 并非完全没有影响，因为 softmax 要求所有概率之和为 1。提高“猫”的概率，必然会挤压其他 token 的概率；交叉熵对 logits 的梯度是：

\[
\frac{\partial L}{\partial z_j}=p_j-\mathbf{1}[j=y]
\]

这意味着一次更新会：

- 提高正确 token 的 logit；
- 按当前概率大小降低其他 token 的 logit。

## 3.1 “我喜欢”后面既有狗又有猫，最优概率是多少？

假设训练集中完全相同的前缀“我喜欢”出现两次：

```text
样本 1：我喜欢 狗
样本 2：我喜欢 猫
```

设模型在这个上下文下：

\[
P(狗)=p,\quad P(猫)=1-p
\]

平均 loss 为：

\[
L(p)=-\frac{1}{2}[\log p+\log(1-p)]
\]

当 \(p=0.5\) 时最小。因此：

```text
P(狗)=0.5
P(猫)=0.5
```

更一般地，如果完全相同上下文中狗占 60%，猫占 40%，交叉熵的总体最优分布就是：

```text
P(狗)=0.6
P(猫)=0.4
```

也就是说，交叉熵不仅让模型记住一个“唯一答案”，还会让模型在无法从上下文区分的情况下逼近训练数据的条件分布。

---

## 4. PPO：限制策略每次不要变化太大

PPO 全称 **Proximal Policy Optimization（近端策略优化）**，由 OpenAI 的 John Schulman、Filip Wolski、Prafulla Dhariwal、Alec Radford 和 Oleg Klimov 于 2017 年提出。原论文为 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。

## 4.1 PPO 解决什么问题？

最朴素的 Policy Gradient 直接提高高 advantage 动作的概率、降低低 advantage 动作的概率。但同一批 rollout 如果被反复优化，当前策略可能已经偏离生成数据时的旧策略；更新太大还可能导致能力崩坏。

PPO 记录旧策略生成 token 的概率，并计算新旧策略概率比：

\[
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
       {\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\]

PPO clipped objective 为：

\[
J_{\mathrm{PPO}}
=\mathbb{E}_t\left[
\min\left(
 r_t(\theta)A_t,
 \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
\]

训练中通常最小化负目标：

\[
L_{\mathrm{policy}}=-J_{\mathrm{PPO}}
\]

直观上：

- 如果新策略只比旧策略变化一点，正常更新；
- 如果概率比变化过大，clip 限制该样本继续推动参数；
- 让每轮策略更新保持在旧策略附近，提高稳定性。

## 4.2 PPO 中 Critic 的作用

PPO 常使用 Actor-Critic 结构：

- Actor：LLM policy，负责生成 token；
- Critic：估计当前状态未来能得到的期望回报 \(V(s_t)\)。

Critic 并不是直接判断“这个 token 正确还是错误”，而是在每个状态预测后续期望 reward。利用 reward 和相邻状态价值，可以通过 TD error / GAE 估计 advantage：

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
\]

\[
A_t^{\mathrm{GAE}}
=\delta_t+(\gamma\lambda)\delta_{t+1}+
(\gamma\lambda)^2\delta_{t+2}+\cdots
\]

因此 PPO 可以得到随 token 位置变化的 \(A_t\)，比整条回答共用同一 reward 更细粒度，但代价是还要训练和保存 Critic。

在 LLM RLHF 中还常加入：

- Critic/value loss；
- entropy bonus；
- 相对 reference model 的 KL penalty；
- reward model 给出的序列奖励。

这些不是 PPO 名称本身的唯一固定配置，但常用于完整训练系统。

---

## 5. GRPO：用同题的一组回答代替 Critic

GRPO 全称 **Group Relative Policy Optimization（组相对策略优化）**，由 DeepSeek 团队的 Zhihong Shao 等人在 2024 年论文 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) 中提出。

## 5.1 GRPO 解决什么问题？

PPO 的 Critic 往往与 policy 模型规模接近，训练和存储成本高。GRPO 的核心思路是：

> 对同一个问题采样一组回答，用组内 reward 分布估计 baseline，不再训练 Critic。

对同一个 prompt 生成 \(G\) 个回答，得到 rewards：

\[
r_1,r_2,\ldots,r_G
\]

标准 GRPO 常对组内 reward 做标准化：

\[
A_i=\frac{r_i-\mu_G}{\sigma_G+\varepsilon}
\]

其中：

\[
\mu_G=\frac{1}{G}\sum_i r_i
\]

这使回答只和“同一道题下的其他回答”比较：

- 高于组平均：正 advantage，提高整条回答中采样 token 的概率；
- 低于组平均：负 advantage，降低其概率；
- 不需要单独的 Critic 网络。

GRPO 仍可以使用 PPO 风格的概率比、clip 和 KL 正则。它与 PPO 最核心的差异不是“有没有 clip”，而是**advantage baseline 的来源不同**。

## 5.2 一个具体例子

同一道 GSM8K 问题生成 4 个回答：

```text
rewards = [1, 0, 1, 0]
mean    = 0.5
```

若只减均值：

```text
advantages = [+0.5, -0.5, +0.5, -0.5]
```

那么：

- 两条正确回答的生成 token 被整体鼓励；
- 两条错误回答的生成 token 被整体抑制。

但要注意一个重要边界：

```text
全部答对 → reward 全为 1 → advantage 全为 0
全部答错 → reward 全为 0 → advantage 全为 0
```

所以 GRPO 训练特别依赖有区分度的题目：模型需要“有时做对、有时做错”。太简单或太难都缺少有效梯度。

---

## 6. nanochat 的算法为什么只是“带引号的 GRPO”

[`../../scripts/chat_rl.py`](../../scripts/chat_rl.py) 开头明确说明：它虽然称为“GRPO”，实际上更接近带组均值 baseline 的 REINFORCE。

它做了四个简化：

1. 删除 trust region，没有相对 reference model 的 KL regularization；
2. 每次使用当前策略新采样的数据立即更新，因此没有 PPO ratio + clip；
3. 使用 DAPO 风格的 token-level normalization；
4. advantage 只用 \(r-\mu\)，不除以标准差。

核心代码是：

```python
rewards = torch.tensor(rewards, dtype=torch.float, device=device)
mu = rewards.mean()
advantages = rewards - mu

logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
num_valid = (targets >= 0).sum().clamp(min=1)
pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
loss = -pg_obj
loss.backward()
```

这里：

- `logp` 形状为 `(B, T)`；
- `advantages` 形状为 `(B,)`；
- `unsqueeze(-1)` 变为 `(B, 1)`；
- PyTorch broadcasting 将每个回答的 advantage 复制到该回答所有 token；
- `*` 是逐元素乘法，不是矩阵乘法。

更完整的源码流程见 [`../rl_GSM8K/rl.md`](../rl_GSM8K/rl.md)。

---

## 7. LLM 训练的统一视角：不管什么范式，本质都是 loss + 反向传播

下面完整保留此前通过飞书发送的整理内容，并在后面补充必要的术语校正。

### 核心结论

无论是预训练、SFT 还是 RL（GRPO/PPO），**从训练机制上看，本质都是同一件事**：

> 对每个样本，计算一个 loss 标量 → `loss.backward()` 反向传播算梯度 → `optimizer.step()` 更新参数。

底层完全一样，区别只在于“loss 是怎么算出来的”。

---

### 7.1 预训练（Pre-training）

**输入**：海量纯文本  
**Loss**：Cross Entropy（每个 token 都算，所有 token 一视同仁）

```text
loss = -mean(log(p_标准答案_token_i))
```

- 有标准答案（由原始文本自动构造）；
- 每个 token 的权重都是 1；
- 信号密集：每个 token 都知道 target 是什么。

---

### 7.2 SFT（监督微调）

**输入**：结构化对话（user + assistant）  
**Loss**：Cross Entropy（只算 assistant 回复部分，prompt 部分 mask 掉）

```text
loss = -mean(log(p_标准答案_token_i))  # 只算 assistant 部分
```

和预训练几乎一样，只是：

- 数据是对话格式，不是纯文本；
- prompt 部分不算 loss（只学怎么回答，不学怎么提问）。

---

### 7.3 RL（以 nanochat 的简化 GRPO 为例）

**输入**：模型自己生成的回答 + 奖励分数  
**Loss**：Advantage-weighted Policy Gradient

```text
advantage = reward - 组内平均 reward
loss = -mean(log(p_采样出来的 token_i) × advantage)
```

和 SFT 的区别：

- 没有逐 token 标准答案，只有奖励分数（例如答对=1，答错=0）；
- 每个 token 的权重不是 1，而是 advantage；
- 信号稀疏：通常只有整个回答结束后才知道结果。

---

### 7.4 三者对比

|  | 预训练 | SFT | RL（nanochat 简化 GRPO） |
|---|---|---|---|
| 数据来源 | 普通文本 | 标准对话 | 当前模型自己生成的回答 |
| 监督信号 | 文本自身产生的 target token | 标准 Assistant token | reward / advantage |
| 每个 token 的权重 | 1 | 1（只算 response） | advantage |
| Loss 公式 | `-log(p_target)` | `-log(p_target)` | `-log(p_sampled) × advantage` |
| 反向传播 | `loss.backward()` | `loss.backward()` | `loss.backward()` |
| 更新参数 | `optimizer.step()` | `optimizer.step()` | `optimizer.step()` |

---

### 7.5 一句话总结

> **不管是预训练、SFT 还是 RL，底层都是同一个东西：算 loss → 反向传播 → 更新参数。**
>
> 区别只是：
>
> - 数据是从哪来的（原始文本、标准回答，还是模型自己生成）；
> - loss 怎么算（和 target token 比，还是根据 reward 加权）；
> - 每个 token 的权重是多少（1、mask 后的 1，还是 advantage）。

就像做饭：不管做中餐还是西餐，锅和火可以相同，但原料、目标味道和评价标准不同。

### 7.6 对原文的准确性补充

此前飞书版本为了突出统一视角，使用了“预训练有标准答案”“RL没有标准答案”的简化说法。严格来说：

- 预训练是**自监督学习**，target 来自文本自身；
- RL 不是无监督学习，reward 就是训练监督信号；
- 标准 PPO/GRPO 的完整 loss 还可能包含 ratio、clip、KL、value loss、entropy 等项；上面的 RL 公式对应的是 nanochat 的极简实现。

---

## 8. PyTorch 自动微分：不是从平均 loss 反推出每个样本

## 8.1 自动微分究竟记录了什么？

假设一个 batch 有 3 个 token loss：

\[
L_1(\theta),L_2(\theta),L_3(\theta)
\]

平均 loss 为：

\[
L(\theta)=\frac{L_1(\theta)+L_2(\theta)+L_3(\theta)}{3}
\]

PyTorch 前向传播时并不是只保存最终数值 `0.65`，而是建立计算图，记录：

- 每个 logit 如何由参数产生；
- 每个 token loss 如何由 logit 和 target 产生；
- 最终 loss 如何由各 token loss 求和、求平均得到。

反向传播应用链式法则：

\[
\nabla_\theta L
=\frac{1}{3}
\left(
\nabla_\theta L_1+
\nabla_\theta L_2+
\nabla_\theta L_3
\right)
\]

所以自动微分不是从均值猜测各项，而是沿仍然保留的计算图分别计算各路径的梯度，再相加。

## 8.2 Batch backward 等价于逐样本 backward 后累积梯度

以下两种方法在满足条件时数学等价。

### 方法 A：整个 batch 一次 backward

```python
optimizer.zero_grad()
loss = model(batch).mean()
loss.backward()
optimizer.step()
```

### 方法 B：逐个样本 backward，只在最后 step

```python
optimizer.zero_grad()
for sample in batch:
    loss_i = model(sample) / len(batch)
    loss_i.backward()  # 只累积梯度，不更新参数
optimizer.step()
```

二者都得到：

\[
\frac{1}{B}\sum_{i=1}^B\nabla_\theta L_i
\]

nanochat 的 gradient accumulation 就使用这个性质：多个 micro-batch 分别 backward，梯度累积完后只执行一次 `optimizer.step()`。

## 8.3 一个重要纠正：不等价于每个样本都 optimizer.step()

下面这种写法通常**不等价**：

```python
for sample in batch:
    optimizer.zero_grad()
    loss_i = model(sample)
    loss_i.backward()
    optimizer.step()  # 每个样本都更新参数
```

原因是第 2 个样本使用的已经是第 1 个样本更新后的参数。对于 AdamW、Muon 等带 momentum/state 的优化器，差异更明显。

所以准确表述应该是：

> **一个 batch 的一次梯度更新，等价于在参数保持不变时逐个样本 backward、按相同比例累积梯度，最后只执行一次 optimizer.step()；不等价于逐样本各自完成一次参数更新。**

BatchNorm、dropout随机性、浮点数加法顺序等也可能带来实现层面的微小差异；nanochat 使用 Transformer/RMSNorm，不涉及 BatchNorm 的 batch 统计问题。

更多梯度下降基础见 [`模型优化与梯度下降.md`](模型优化与梯度下降.md)。

---

## 9. 最终心智模型

可以把三类训练统一成下面的形式：

\[
L(\theta)
=-\frac{1}{N}\sum_t
w_t\log p_\theta(a_t\mid s_t)
\]

区别集中在 \(a_t\) 与 \(w_t\)：

| 训练方式 | \(a_t\) 是什么 | \(w_t\) 是什么 |
|---|---|---|
| 预训练 | 原始文本中的下一个 token | 通常为 1 |
| SFT | 标准 Assistant 回答的下一个 token | Assistant 位置为 1，其余为 0 |
| nanochat RL | 当前策略实际采样的 token | 该回答的 advantage |
| PPO | 旧策略 rollout 中采样的 token | token-level advantage，并受 ratio/clip 约束 |
| GRPO | 同题一组 rollout 中采样的 token | 组内相对 advantage，并可受 ratio/clip/KL 约束 |

这就是理解 RL 的入口：**训练底座仍然是概率、loss、自动微分和梯度下降；RL 真正增加的是 rollout、reward、advantage、探索和信用分配。**

---

## 参考资料

- [Proximal Policy Optimization Algorithms, Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, Shao et al., 2024](https://arxiv.org/abs/2402.03300)
- nanochat RL 实现：[`../../scripts/chat_rl.py`](../../scripts/chat_rl.py)
- nanochat GPT loss：[`../../nanochat/gpt.py`](../../nanochat/gpt.py)
- nanochat SFT 学习笔记：[`../stf/sft.md`](../stf/sft.md)
