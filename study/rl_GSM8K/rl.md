# nanochat：在 GSM8K 上学习强化学习

本文以 [`scripts/chat_rl.py`](../../scripts/chat_rl.py) 为主线，解释 nanochat 如何在 GSM8K 数学题上执行强化学习。

阅读目标不是只知道代码“做了什么”，而是能回答我们讨论时反复追问的几个问题：

1. SFT 与 RL 到底哪里一样、哪里不同？
2. RL 没有逐 token 标准答案，loss 从哪里来？
3. `(B, T)` 的 `logp` 分别表示什么？
4. 不同长度的回答如何组成 batch？
5. advantage 如何广播给每一个 token？
6. `loss.backward()` 最终怎样提高正确回答的概率、降低错误回答的概率？

相关基础概念：

- [对比 SFT 与强化学习来学习 RL](../threshold_concept/对比SFT与强化学习来学习RL.md)
- [CoT、RL 与能力涌现](../threshold_concept/COT、RL与能力涌现.md)
- [nanochat SFT 机制简介](../stf/sft.md)

---

## 1. nanochat RL 介绍

nanochat 在 [`scripts/chat_rl.py`](../../scripts/chat_rl.py) 中实现了一个非常精简的 GSM8K 强化学习流程。

入口命令：

```bash
# 单 GPU
python -m scripts.chat_rl

# 8 GPU
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
```

训练任务是 GSM8K：给模型一道小学数学应用题，让模型生成推理和最终答案。程序从回答中抽取 `#### 数字`，与数据集标准数值比较：

```text
预测数值 == 标准数值 → reward = 1.0
否则                 → reward = 0.0
```

奖励实现位于 [`tasks/gsm8k.py`](../../tasks/gsm8k.py)：

```python
def reward(self, conversation, assistant_response):
    is_correct = self.evaluate(conversation, assistant_response)
    return float(is_correct)
```

## 1.1 它为什么被称为“带引号的 GRPO”？

`chat_rl.py` 文件开头主动说明：它虽然写作 “GRPO”，实际上比标准 GRPO 更简单，更像 **REINFORCE with group-mean baseline**。

相比标准 PPO/GRPO，它删除或简化了：

1. 没有 reference model KL regularization；
2. 没有 PPO probability ratio 和 clipping；
3. 使用 token-level normalization，而不是先对每个序列独立平均；
4. advantage 只使用 \(r-\mu\)，不做 \((r-\mu)/\sigma\) 标准化；
5. 没有 Critic/value network。

所以学习这段代码时应建立准确定位：

> 它不是生产级完整 GRPO，而是用最少代码展示“rollout → reward → relative advantage → policy-gradient loss → backward”的核心机制。

## 1.2 它从什么模型开始？

```python
model, tokenizer, meta = load_model(
    "sft", device, phase="eval",
    model_tag=args.model_tag,
    step=args.model_step,
)
```

RL 不从随机模型开始，而是加载 SFT checkpoint：

```text
预训练 base model
      ↓
SFT chat model
      ↓
GSM8K RL
```

原因是 RL 需要模型至少能：

- 理解问题；
- 生成可读回答；
- 遵循 `#### 答案` 格式；
- 偶尔找到正确解法。

如果同一道题采样很多次仍全部错误，组内 advantage 全为 0，就缺乏有效学习信号。

---

## 2. SFT 与 RL 的本质区别

详细说明见 [对比 SFT 与强化学习来学习 RL](../threshold_concept/对比SFT与强化学习来学习RL.md)，这里只保留理解源码所需的最小心智模型。

## 2.1 共同点

SFT 和 RL 都要把训练目标写成一个可微的标量 loss：

```python
loss.backward()
optimizer.step()
```

二者使用同一个 GPT forward、同一个 PyTorch autograd 和同一类优化器。

## 2.2 SFT 的目标

SFT 有标准 Assistant 回答，每个有效位置都有 target token：

\[
L_{\mathrm{SFT}}
=-\frac{1}{N}\sum_t
\log p_\theta(y_t\mid x,y_{<t})
\]

模型要提高**标准答案 token** 的概率。

## 2.3 RL 的目标

RL 没有唯一标准回答。模型先自己采样 token，整个回答完成后再得到 reward：

\[
L_{\mathrm{RL}}
=-\frac{1}{N}\sum_t
A\log p_\theta(a_t\mid s_t)
\]

模型要根据 advantage：

- 提高高 reward 回答中**实际采样 token** 的概率；
- 降低低 reward 回答中**实际采样 token** 的概率。

最核心的区别不是“是否使用 token 概率”——两者都使用概率和 log probability。区别是：

```text
SFT：取标准 target token 的 logp，权重通常为 1
RL：取模型采样 token 的 logp，权重为 advantage
```

---

## 3. nanochat RL 完整流程

整体流程如下：

```text
加载 SFT checkpoint
    ↓
从 GSM8K 读取一道问题
    ↓
删除标准 Assistant 答案，只保留问题和 assistant_start
    ↓
当前模型为同一问题生成 16 条回答
    ↓
抽取每条回答的最终数字，计算 0/1 reward
    ↓
组内 reward 减均值，得到 advantage
    ↓
padding 到相同长度，构造 inputs / targets / mask
    ↓
重新计算每个采样 token 的 logp
    ↓
logp × advantage，按有效 token 归一化
    ↓
取负得到 loss
    ↓
backward 累积梯度
    ↓
optimizer.step 更新模型
    ↓
周期性计算 GSM8K pass@k 并保存 checkpoint
```

下面逐步展开。

---

## 3.1 初始化计算环境

```python
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
```

这部分与预训练/SFT 类似：

- 自动选择 CUDA、MPS 或 CPU；
- 使用 `torchrun` 时初始化分布式环境；
- 每个 rank 处理不同题目；
- rank 0 负责主要日志和 checkpoint。

W&B 项目名是 `nanochat-rl`。

---

## 3.2 加载 SFT 模型和 Engine

```python
model, tokenizer, meta = load_model("sft", device, phase="eval", ...)
engine = Engine(model, tokenizer)
```

两个对象职责不同：

- `model`：重新计算 rollout token 的概率并执行 backward；
- `engine`：自回归采样回答。

同一个 policy 同时承担 rollout 和训练，因此这段代码是 on-policy 风格：采样后立即使用这些样本更新当前模型。

---

## 3.3 准备 GSM8K 训练集和测试集

```python
train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
```

默认：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `num_epochs` | 1 | 遍历 GSM8K 训练问题的轮数 |
| `examples_per_step` | 16 | 一次 optimizer step 跨所有 rank 使用多少道题 |
| `num_samples` | 16 | 每道题生成多少条回答 |
| `device_batch_size` | 8 | 一次 forward/generate 最多处理多少条序列 |
| `max_new_tokens` | 256 | 每条回答最多生成多少 token |
| `temperature` | 1.0 | rollout 采样温度 |
| `top_k` | 50 | 只从概率最高的前 50 个 token 中采样 |

一次 optimizer step 默认涉及：

```text
16 道题 × 每题 16 条回答 = 256 条生成序列
```

但为了避免 OOM，这些序列会分批生成和 forward。

---

## 3.4 把标准答案从输入中删除

GSM8K 的 `conversation` 同时包含 user 问题和标准 Assistant 答案。RL rollout 时不能把答案泄露给模型：

```python
tokens = tokenizer.render_for_completion(conversation)
prefix_length = len(tokens)
```

`render_for_completion()` 会：

1. 深复制 conversation；
2. 删除最后一条标准 Assistant message；
3. 渲染剩余上下文；
4. 在末尾追加 `<|assistant_start|>`。

因此输入类似：

```text
<|bos|>
<|user_start|>GSM8K问题<|user_end|>
<|assistant_start|>
```

模型从这里开始自己生成回答。

---

## 3.5 对同一道题生成多条 rollout

```python
generated_token_sequences_batch, masks_batch = engine.generate_batch(
    tokens,
    num_samples=args.device_batch_size,
    max_tokens=args.max_new_tokens,
    temperature=args.temperature,
    top_k=args.top_k,
    seed=seed,
)
```

默认每道题生成 16 条回答，但 `device_batch_size=8`，所以分两次生成，避免显存不足。

Rollout 必须具有一定随机性：如果每次都贪心解码，同一道题会得到相同回答，组内无法形成比较。`temperature=1.0` 和 `top_k=50` 让模型探索不同推理轨迹。

Engine 同时返回 `mask`，用于区分：

- prompt token：不训练；
- 模型真正采样的 Assistant token：训练；
- 工具系统强制写入的 token：不训练；
- 后续 padding：不训练。

---

## 3.6 计算每条回答的 reward

```python
generated_tokens = sample_tokens[prefix_length:]
generated_text = tokenizer.decode(generated_tokens)
reward = train_task.reward(conversation, generated_text)
```

`tasks/gsm8k.py` 使用正则：

```python
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
```

分别抽取：

- 数据集标准回答中的 `#### number`；
- 模型回答中的 `#### number`。

二者字符串归一化后相等：

```text
reward = 1.0
```

否则：

```text
reward = 0.0
```

这是 outcome reward：只判断最终数字，不评价中间推理过程是否正确、简洁或优雅。

### 例子

同一道题的 4 条回答：

```text
回答1：#### 32 → 正确 → reward=1
回答2：#### 30 → 错误 → reward=0
回答3：#### 32 → 正确 → reward=1
回答4：没有 #### → 无法提取 → reward=0
```

---

## 3.7 处理不同长度：T 取本组最长序列

不同回答长度不同，不能直接堆成矩阵。代码先取：

```python
max_length = max(len(seq) for seq in generated_token_sequences)
```

短序列使用 `<|assistant_end|>` 补齐：

```python
padded_generated_token_sequences = [
    seq + [assistant_end] * (max_length - len(seq))
    for seq in generated_token_sequences
]
```

mask 同时补 0：

```python
padded_masks = [
    mask + [0] * (max_length - len(mask))
    for mask in masks
]
```

假设 3 条序列长度分别为 5、3、4：

```text
seq1: [a, b, c, d, e]
seq2: [f, g, h, PAD, PAD]
seq3: [i, j, k, l, PAD]
```

于是：

```text
B = 3
序列总长度 = 5
```

移位后用于预测的时间步数是 4：

```python
inputs  = ids[:, :-1]
targets = ids[:, 1:]
```

所以严格说：

```text
ids 的形状     = (B, L)
inputs/targets = (B, T)，其中 T = L - 1
```

我们讨论中把 `T` 说成“batch 最长序列长度”是直觉化表达；对应到 `logp` 代码时，`T` 实际是移位后的预测位置数。

padding、prompt 和非采样 token 的 target 被设为 `-1`：

```python
targets[mask_ids[:, 1:] == 0] = -1
```

`GPT.forward()` 的交叉熵设置 `ignore_index=-1`，所以这些位置产生 0 loss，不影响梯度。

---

## 3.8 计算 advantage

```python
rewards = torch.tensor(rewards, dtype=torch.float, device=device)
mu = rewards.mean()
advantages = rewards - mu
```

以上面的 rewards 为例：

```text
rewards = [1, 0, 1, 0]
mu      = 0.5
advantages = [0.5, -0.5, 0.5, -0.5]
```

解释：

- 回答 1、3 高于组平均，整体提高概率；
- 回答 2、4 低于组平均，整体降低概率。

### 为什么减均值？

如果直接使用 0/1 reward：

```text
正确回答 → 有正向梯度
错误回答 → reward=0，完全没有梯度
```

减去均值后，错误回答得到负 advantage，可以主动降低其概率，并通常有助于稳定组内相对比较、降低梯度波动。需要注意：这里的组均值由同一批采样回答共同计算，并非经典理论中与当前动作独立的 baseline；其估计性质和偏差需要按 GRPO 的具体目标分析，不能简单等同于“不改变期望梯度”的普通 baseline。

### nanochat 与标准 GRPO 的区别

标准 GRPO 常用：

\[
A_i=\frac{r_i-\mu}{\sigma+\epsilon}
\]

nanochat 只用：

\[
A_i=r_i-\mu
\]

因此它保留方向，但不同组的梯度尺度会随 reward 方差变化。

---

## 3.9 重新计算采样 token 的 log probability

关键代码：

```python
logp = -model(
    inputs,
    targets,
    loss_reduction='none'
).view_as(inputs)
```

GPT 内部调用：

```python
F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=-1,
    reduction='none',
)
```

Cross entropy 对每个 target token 返回：

\[
\mathrm{NLL}_t=-\log p_\theta(a_t\mid s_t)
\]

代码再取负：

\[
\mathrm{logp}_t=
\log p_\theta(a_t\mid s_t)
\]

所以 `logp` 的形状是：

```text
(B, T)
```

含义是：

- `B`：当前 forward 中有多少条回答；
- `T`：每条回答移位后有多少个预测位置；
- `logp[b, t]`：模型在第 `b` 条回答、第 `t` 个位置，给**实际采样 token** 分配的 log probability。

### 只关心采样 token，还是整个词表？

Loss 显式读取的是采样 token 的 logp，而不是把词表中所有 token 的概率分别写进 loss。

但其他 token 仍间接参与：softmax 概率和为 1，提高采样 token 的概率会挤压其他 token；梯度也会作用到整个 logits 分布。

---

## 3.10 Advantage broadcasting：不是矩阵乘法

代码：

```python
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
```

假设：

```text
logp.shape       = (2, 3)
advantages.shape = (2,)
```

先：

```python
advantages.unsqueeze(-1)
```

形状变为 `(2, 1)`：

```text
[[+0.75],
 [-0.25]]
```

PyTorch broadcasting 在逻辑上扩展成：

```text
[[+0.75, +0.75, +0.75],
 [-0.25, -0.25, -0.25]]
```

再和 `(2, 3)` 的 `logp` **逐元素相乘**。这里的 `*` 不是矩阵乘法；矩阵乘法通常使用 `@` 或 `torch.matmul()`。

因此同一回答内所有有效 token 共用该回答的 advantage。

---

## 3.11 Policy-gradient objective 和 loss

代码首先最大化：

```python
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
```

对应：

\[
J(\theta)
=\sum_{i,t}A_i
\log p_\theta(a_{i,t}\mid s_{i,t})
\]

然后按有效 token 数归一化：

```python
num_valid = (targets >= 0).sum().clamp(min=1)
pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
```

由于优化器执行最小化，取负号：

```python
loss = -pg_obj
```

即：

\[
L_{\mathrm{nanochat\ RL}}
=-\frac{1}{N_{\mathrm{valid}}}
\sum_{i,t}A_i
\log p_\theta(a_{i,t}\mid s_{i,t})
\]

### 梯度方向为什么正确？

对一个采样 token：

- \(A_i>0\)：最小化 `-A_i logp` 会提高该 token 的概率；
- \(A_i<0\)：最小化 `-A_i logp` 会降低该 token 的概率；
- target 为 `-1`：交叉熵输出 0，不产生训练贡献。

这和 SFT 的结构非常接近：

```text
SFT：-1 × logp(标准 token)
RL： -A × logp(采样 token)
```

---

## 3.12 Backward、梯度累积与 optimizer step

```python
loss.backward()
```

每道题的 rollout 可能还要分成多个 `device_batch_size` pass；一次 optimizer step 也会处理多道题。代码在这些循环中持续 backward，梯度累积到参数的 `.grad` 中。

只有所有题和所有 pass 都完成后才：

```python
optimizer.step()
model.zero_grad(set_to_none=True)
```

因此一次参数更新使用的是这一整个 optimization step 中所有 rollout 的平均梯度。

这里应用了自动微分和梯度线性叠加，详细解释见[对比 SFT 与强化学习来学习 RL：自动微分](../threshold_concept/对比SFT与强化学习来学习RL.md#8-pytorch-自动微分不是从平均-loss-反推出每个样本)。

需要强调：

> 多个样本一次更新，等价于参数保持不变时逐个 backward 并累积梯度，最后只 step 一次；不等价于每个样本分别执行一次 optimizer.step()。

---

## 3.13 学习率调度

RL 使用和 SFT/预训练相同的 Muon + AdamW 参数分组，但学习率设置更保守：

```python
for group in optimizer.param_groups:
    group["lr"] *= args.init_lr_frac
```

默认：

```text
init_lr_frac = 0.05
```

也就是只使用基础学习率的 5%。原因是 RL 从已有 SFT 模型继续训练，过大的更新可能破坏已有能力。

学习率从初始值线性下降到 0：

\[
\mathrm{lrm}(t)=1-\frac{t}{\mathrm{num\_steps}}
\]

```python
group["lr"] = group["initial_lr"] * lrm
```

weight decay 默认为 0。

---

## 3.14 为什么没有 PPO ratio、clip 和 KL？

代码注释给出的理由是：当前实现采样后立即训练，属于 on-policy，所以不再加入 PPO ratio + clip。

更完整地说：

- 标准 PPO 通常固定一批旧策略 rollout，并对其做多个 optimization epoch；随着参数更新，新旧策略概率会偏离，ratio/clip 用于控制更新幅度；
- nanochat 每批 rollout 只做一次当前更新，减少了旧数据重复使用导致的偏离；
- 项目以教学和极简为目标，直接省略这些稳定机制。

但“on-policy”并不意味着任何系统都绝对不需要 KL 或更新约束。大规模训练中，即使采样很新，单步更新过大、reward hacking 或通用能力退化仍可能发生。

---

## 3.15 评估：Pass@k

训练中周期性在 GSM8K test split 上生成多条回答：

```python
for k in range(1, args.device_batch_size + 1):
    passk[k - 1] = sum(
        any(o["is_correct"] for o in r["outcomes"][:k])
        for r in records
    )
```

Pass@k 的含义：

> 对每道题生成前 k 个候选，只要至少一个正确，该题就算通过。

例如 100 道题：

```text
每题生成 8 个回答
其中 70 道至少有一个正确
→ Pass@8 = 0.70
```

Pass@1 衡量单次回答能力；Pass@k 同时反映模型是否保留了足够的候选多样性和探索能力。

---

## 3.16 保存 checkpoint

训练中按 `save_every` 或最后一步保存：

```text
chatrl_checkpoints/d{depth}/
```

保存内容：

- 模型参数；
- 模型配置；
- 不保存 optimizer state。

所以当前实现主要面向一次性实验流程，不具备和预训练同等级别的完整断点恢复能力。

---

## 4. 用一个完整数字例子串起 loss

假设同一道题生成两条回答，每条有效生成 token 数都是 3：

```text
回答 A：正确，reward = 1
回答 B：错误，reward = 0
```

组均值：

\[
\mu=0.5
\]

advantage：

```text
A_A = +0.5
A_B = -0.5
```

模型重新计算实际采样 token 的 logp：

```text
logp_A = [-0.2, -0.4, -0.3]
logp_B = [-0.1, -0.5, -0.8]
```

逐元素乘 advantage：

```text
A: [-0.10, -0.20, -0.15]
B: [+0.05, +0.25, +0.40]
```

目标和：

```text
pg_obj = 0.25
```

有效 token 共 6 个：

\[
J=0.25/6
\]

训练 loss：

\[
L=-J
\]

不能只根据当前 loss 的正负判断训练好坏；关键是它对参数的梯度方向：

- A 中 3 个 token 的概率被提高；
- B 中 3 个 token 的概率被降低。

不同题、不同回答的梯度最终在参数空间相加，optimizer 再执行一次更新。

---

## 5. nanochat RL 能学到什么，不能学到什么？

## 5.1 能学到的

如果模型在某类数学题上偶尔能生成正确轨迹，训练会提高这些轨迹及相似生成模式的概率，例如：

- 正确列出中间算式；
- 使用足够的推理步骤；
- 输出 `#### number` 格式；
- 在相似题目上更常走向正确答案。

## 5.2 无法精确知道的

因为 reward 只判断最终答案：

- 不知道具体哪一步推理正确；
- 不知道是否碰巧猜对；
- 不知道中间过程是否存在逻辑错误；
- 不知道某些冗长 token 是否有用；
- 正确回答的所有采样 token 都拿同一 advantage；
- 错误回答中的正确局部步骤也会被一起抑制。

这就是 outcome reward 的信用分配局限。

## 5.3 数据区分度限制

若 16 条回答全部正确或全部错误：

```text
advantages = rewards - rewards.mean() = 全 0
```

这一题不会提供梯度。因此提高 RL 效果不仅要改算法，也要：

- 选择位于模型能力边界的问题；
- 调整采样温度和样本数；
- 提高 verifier 可靠性；
- 必要时引入更细粒度过程奖励；
- 防止模型只学格式或 reward hacking。

---

## 6. nanochat RL 与完整 PPO/GRPO 对比

| 维度 | nanochat RL | 标准 PPO | 标准 GRPO |
|---|---|---|---|
| 起点 | SFT checkpoint | policy + 常见 reference/reward/value model | policy + 常见 reference/reward model |
| Advantage | 同题 reward 减均值 | Critic + GAE | 同题组内 reward 标准化 |
| Critic | 无 | 通常有 | 无 |
| Ratio | 无 | 有 | 通常有 |
| Clip | 无 | 有 | 通常有 |
| KL | 无 | LLM RLHF 中常有 | 常见实现可有 |
| 归一化 | 全部有效 token | 依实现而定 | 原论文/实现依配置而定 |
| Reward | GSM8K 最终答案 0/1 | 可来自环境或 reward model | 常用于可验证 reward |
| 定位 | 教学型极简 REINFORCE | 稳定通用策略优化 | 省去 Critic 的组相对优化 |

---

## 7. 最值得记住的源码对应关系

| 概念 | nanochat 代码 |
|---|---|
| 加载 SFT 模型 | `scripts/chat_rl.py:74` |
| 删除标准答案、准备 completion | `nanochat/tokenizer.py:367-385` |
| 同题多次采样 | `scripts/chat_rl.py:99-115` |
| GSM8K 0/1 reward | `tasks/gsm8k.py:87-117` |
| padding 与 mask | `scripts/chat_rl.py:128-140` |
| 组均值 advantage | `scripts/chat_rl.py:141-144` |
| 逐 token NLL / logp | `nanochat/gpt.py:416-480`、`scripts/chat_rl.py:263-264` |
| advantage broadcasting | `scripts/chat_rl.py:266` |
| token-level normalization | `scripts/chat_rl.py:267-269` |
| loss 与 backward | `scripts/chat_rl.py:270-273` |
| optimizer step | `scripts/chat_rl.py:296-301` |
| Pass@k | `scripts/chat_rl.py:224-243` |

---

## 8. 最终总结

nanochat 的 RL 可以用下面六步记住：

1. **出题**：从 GSM8K 取一道题；
2. **探索**：当前 SFT 模型生成 16 条回答；
3. **判分**：最终数字正确得 1，错误得 0；
4. **相对比较**：reward 减组均值成为 advantage；
5. **构造 loss**：每个采样 token 的 `logp × advantage`，对有效 token 求平均并取负；
6. **更新策略**：backward + optimizer.step，提高正确轨迹概率、降低错误轨迹概率。

它展示了强化学习最核心的转换：

```text
最终结果的 reward
    ↓
回答级 advantage
    ↓
广播到生成 token
    ↓
advantage-weighted log probability
    ↓
可反向传播的 loss
    ↓
模型参数更新
```

也要同时记住其局限：它没有 Critic、PPO ratio、clip、KL 和过程奖励，所以最适合学习核心机制，而不能代表完整生产级 RL 系统。
