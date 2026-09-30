# nanochat SFT 机制简介

SFT（Supervised Fine-Tuning，监督微调）位于预训练之后。预训练让模型掌握语言规律和通用知识，SFT 则使用带有标准答案的对话数据，教模型如何扮演 Assistant：理解用户指令、按照约定格式回答、调用工具，并在特定任务上给出预期输出。

nanochat 的 SFT 没有更换模型结构，也没有引入新的语言建模目标。它仍然使用同一个 GPT，通过 next-token prediction 训练。关键变化集中在三处：训练数据变成结构化对话；模型从预训练 checkpoint 开始；loss 只计算 Assistant 应当生成的 token。

## 1. SFT 的目标

预训练面对的是普通文本，其目标近似为：

\[
L_{\text{pretrain}}
=-\sum_t \log p_\theta(x_t\mid x_{<t})
\]

SFT 面对的是包含 User 和 Assistant 的对话，但只在监督掩码为 1 的位置计算损失：

\[
L_{\text{SFT}}
=-\sum_{t:\,m_t=1}\log p_\theta(y_t\mid x_{<t})
\]

用户问题仍然会进入模型的上下文，模型可以通过 attention 读取它；只是用户问题本身不作为预测目标。模型真正被要求模仿的是标准 Assistant 回答。

nanochat 默认通过 SFT 教授以下能力：

- 通用对话和指令遵循；
- nanochat 的身份、能力和边界；
- MMLU 选择题格式；
- GSM8K 数学推理；
- Python 工具调用及工具返回后的继续作答；
- 单词拼写和字符计数。

因此，SFT 的作用不是简单地“再学一遍知识”，而是把已经具备语言能力的 base model 转换成能按照对话协议工作的 chat model。它不使用 reward，也不进行偏好优化；这些属于后续 RL 阶段。

## 2. 训练数据概览

nanochat 默认将 SmolTalk、身份对话、MMLU、GSM8K、SimpleSpelling 和 SpellingBee 混合成一个训练集。同一个任务被重复加入 mixture，就相当于对它进行过采样。当前默认配置共有 **1,071,759 条 conversation**：

| 数据集 | 主要用途 | 默认样本数 | 按行数占比 |
|---|---|---:|---:|
| SmolTalk | 通用对话和指令遵循 | 460,341 | 42.95% |
| Identity ×2 | nanochat 身份和能力边界 | 2,000 | 0.19% |
| MMLU ×3 | 知识问答和选择题格式 | 299,526 | 27.95% |
| GSM8K ×4 | 数学推理和工具使用 | 29,892 | 2.79% |
| SimpleSpelling | 将单词拆成字符 | 200,000 | 18.66% |
| SpellingBee | 字符计数和工具验证 | 80,000 | 7.46% |

这里的比例按 conversation 行数计算，不等于实际 token 占比，因为不同样本的长度差异很大。数据定义位于 [`scripts/chat_sft.py`](../../scripts/chat_sft.py)，混合和固定随机种子打乱的逻辑位于 [`tasks/common.py`](../../tasks/common.py)。

各类数据的 10 条真实样本见[附录 A：训练样本](#appendix-samples)。

## 3. 对话渲染与监督掩码

每条 conversation 会由 `render_conversation()` 转换成 token 序列。一个普通问答大致会变成：

```text
<|bos|>
<|user_start|>用户问题<|user_end|>
<|assistant_start|>标准回答<|assistant_end|>
```

与此同时，tokenizer 还会生成一个等长的 loss mask：

```text
内容                         mask
<|bos|>                       0
<|user_start|>                0
用户问题                       0
<|user_end|>                  0
<|assistant_start|>           0
Assistant 回答                1
<|assistant_end|>             1
```

对于工具调用，监督规则更加细致：

```text
Assistant 解释文本            1
<|python_start|>              1
Python 表达式                  1
<|python_end|>                1
<|output_start|>              0
Python 执行结果                0
<|output_end|>                0
Assistant 最终回答             1
```

模型需要学会何时生成 Python 调用以及调用什么表达式，但 Python 的执行结果由外部解释器提供，不应该要求模型猜测。因此，工具输出的 mask 是 0；模型只需读取这个结果并继续作答。

在构造 `targets` 时，mask 为 0 的位置会被改成 `-1`。GPT 的交叉熵使用 `ignore_index=-1`，这些位置不会产生 loss。相关实现位于 [`nanochat/tokenizer.py`](../../nanochat/tokenizer.py)、[`scripts/chat_sft.py`](../../scripts/chat_sft.py) 和 [`nanochat/gpt.py`](../../nanochat/gpt.py)。

## 4. SFT 与预训练的关系

SFT 与预训练共享绝大部分训练基础设施：

- 同一个 GPT Transformer 和 tokenizer；
- 同一个 causal cross-entropy；
- Muon 与 AdamW 组合优化器；
- DDP、梯度累积和混合精度；
- `torch.compile`、validation BPB、W&B 和 checkpoint。

两者的核心差异如下：

| 维度 | 预训练 | SFT |
|---|---|---|
| 模型起点 | 随机初始化 | 加载 base checkpoint |
| 数据 | ClimbMix 普通文本 | 结构化 conversation mixture |
| 监督范围 | 基本覆盖所有文本 token | 只覆盖 Assistant 应生成的 token |
| 输入格式 | BOS + 普通文本 | User、Assistant、Python 等特殊 token |
| 数据打包 | 空间不足时可裁剪文档 | 尽量保留完整 conversation，剩余位置 padding |
| 训练长度 | 由 FLOPs 或 token/parameter 比例决定 | 默认消费完整 mixture 一轮 |
| 学习率 | 根据规模和 batch size 计算 | 继承预训练配置，再乘 `init_lr_frac` |
| Weight decay | 训练中逐渐衰减至 0 | 固定为 0 |
| 评估 | CORE 和普通文本续写 | ChatCORE、MMLU、GSM8K、HumanEval 等 |
| 恢复能力 | 支持中途保存和 resume | 当前实现主要在训练结束时保存 |

两种训练最终都会执行同一种基本更新：

```python
loss = model(x, y)
loss.backward()
optimizer.step()
```

区别主要来自 `x/y` 如何构造，以及哪些 `y` 被设成 `-1`。

## 5. SFT 的完整流程

### 5.1 准备输入产物

SFT 依赖已经训练好的 tokenizer 和 base checkpoint。此外还需要身份对话文件、Hugging Face 数据集和拼写任务使用的英文单词表。标准执行入口可参考 [`runs/speedrun.sh`](../../runs/speedrun.sh)。

### 5.2 初始化训练环境

脚本检测 CUDA、CPU 或 MPS，初始化 DDP 和 W&B，选择计算精度，并检查 Flash Attention 是否可用。

### 5.3 加载预训练模型

```python
model, tokenizer, meta = load_model(
    "base", device, phase="train"
)
```

SFT 不重新创建随机模型，而是从 base checkpoint 继续训练。它还会继承 `max_seq_len`、batch size 和各参数组的学习率。

### 5.4 初始化优化器

SFT 继续使用 Muon + AdamW。默认情况下，它还会加载预训练优化器的 momentum 等状态，但重新设置适用于 SFT 的学习率。初始学习率默认是继承值的 `0.8`，weight decay 为 0。

### 5.5 构建并打乱数据 mixture

所有任务被放入 `TaskMixture`。它先建立 `(task_index, local_index)` 映射，再使用固定种子 42 打乱，使不同任务分散在整个训练过程中。DDP 下，每个 rank 读取不同的 conversation。

### 5.6 渲染 conversation

tokenizer 添加角色和工具特殊 token，并生成 loss mask。若第一条消息是 system message，当前实现会把它合并到随后第一条 user message。过长 conversation 最多保留 2048 个 token。

### 5.7 Best-fit 打包 batch

每个训练行容量为 `max_seq_len + 1`。dataloader 从缓冲区里反复选择能够完整放入剩余空间的最长 conversation。没有 conversation 能继续放入时，就用 BOS token padding，并将 padding target 屏蔽。

随后通过移位得到：

```python
inputs = batch[:, :-1]
targets = batch[:, 1:]
```

User、工具输出和 padding 对应的 target 会被设为 `-1`。

### 5.8 评估与训练

训练开始前、默认每 200 step 以及训练结束时，脚本会计算 validation BPB 和 ChatCORE。ChatCORE 综合评估 ARC-Easy、ARC-Challenge、MMLU、GSM8K、HumanEval 和 SpellingBee。

每个 optimization step 依次执行：

1. 运行一个或多个 micro-batch；
2. 对 Assistant token 计算 loss；
3. backward 并累积梯度；
4. 根据数据消费进度调整学习率；
5. 调整 Muon momentum；
6. 执行 `optimizer.step()` 并清空梯度；
7. 记录 loss、吞吐、MFU 和训练进度。

### 5.9 停止并保存

默认 `--num-iterations=-1`，即训练到整个 mixture 被消费一遍。也可以显式指定 optimization step 数进行快速实验。

结束时，脚本进行最终评估，并把模型、各 rank 的 optimizer shard、模型配置和运行参数写入：

```text
chatsft_checkpoints/d{depth}/
```

整个过程可以概括为：

```text
Base checkpoint
    ↓
构建并打乱对话任务 mixture
    ↓
添加角色/工具特殊 token 和 loss mask
    ↓
Best-fit packing + padding
    ↓
对 Assistant token 做 next-token prediction
    ↓
Validation BPB + ChatCORE
    ↓
保存 SFT checkpoint
```

<a id="appendix-samples"></a>

## 附录 A：10 条真实训练样本

下面的样本直接来自 nanochat 当前 SFT 使用的数据类。为了展示工具数据的真实结构，`text`、`python` 和 `python_output` 部分被明确标出。

### A.1 SmolTalk：文本改写

```text
User:
Transform this sentence: "The cat sat on the mat"
The cat sat on the mat

Assistant:
On the mat sat the cat.
```

### A.2 SmolTalk：知识问答

```text
User:
Q: Q: A system of glands secretes what chemical messenger molecules into the blood?   A:
The answer is:

Assistant:
Hormones
```

### A.3 Identity：语言能力

```text
User:
hey! quick question, do you speak any languages other than English?

Assistant:
I can understand and respond in several different languages, but I work best in English.
This is because I was trained primarily on the FineWeb-edu dataset, which consists mostly
of English text, and my tokenizer was optimized for it. While I'll do my best to help you
in other languages, you will definitely get the highest quality and most accurate responses
if we stick to English.
```

### A.4 Identity：记忆能力边界

```text
User:
hey! quick question, will you remember what we talked about if i close this window and
come back tomorrow?

Assistant:
I won't be able to remember our conversation if you leave and come back. I don't have a
persistent memory between sessions, which means every time we start a new chat, it's like
a blank slate for me. I'm a minimal, open-source model called nanochat, and while I can help
you with questions right now, I can't store information about you or our past interactions
once the window is closed.
```

### A.5 MMLU：常识选择题

```text
User:
Multiple Choice question: Deserts are generally
- sweltering=A
- cold=B
- tepid=C
- frigid=D

Respond only with the letter of the correct answer.

Assistant:
A
```

### A.6 MMLU：化学选择题

```text
User:
Multiple Choice question: Sodium, Na, is in the same group as
- Ne.=A
- Mg.=B
- Ca.=C
- K.=D

Respond only with the letter of the correct answer.

Assistant:
D
```

### A.7 GSM8K：数学推理

```text
User:
Stacy has 2 more than triple as many berries as Steve. Steve has one half as many berries
as Skylar. If Skylar has 20 berries, how many berries does Stacy have?

Assistant (text):
Steve has (1/2)(20)=10 berries
Stacy has 2+3(10)=32 berries
#### 32
```

### A.8 GSM8K：数学推理和 Python 工具调用

```text
User:
Mimi picked up 2 dozen seashells on the beach. Kyle found twice as many shells as Mimi
and put them in his pocket. Leigh grabbed one-third of the shells that Kyle found.
How many seashells did Leigh have?

Assistant (text):
Mimi has 2 x 12 =

Assistant (python):
2*12

Python output:
24

Assistant (text):
24 sea shells.
Kyle has 24 x 2 =

Assistant (python):
24*2

Python output:
48

Assistant (text):
48 sea shells.
Leigh has 48 / 3 =

Assistant (python):
48/3

Python output:
16

Assistant (text):
16 sea shells.
#### 16
```

### A.9 SimpleSpelling：拼写

```text
User:
Spell the word: baggers

Assistant:
baggers:b,a,g,g,e,r,s
```

### A.10 SpellingBee：字符计数和工具验证

```text
User:
count the number of "s" in 'nonclassifiable'?

Assistant (text):
We are asked to find the number 's' in the word 'nonclassifiable'. Let me try a manual
approach first.

First spell the word out:
nonclassifiable:n,o,n,c,l,a,s,s,i,f,i,a,b,l,e

Then count the occurrences of 's':
1:n
2:o
3:n
4:c
5:l
6:a
7:s hit! count=1
8:s hit! count=2
9:i
10:f
11:i
12:a
13:b
14:l
15:e

This gives us 2.

Let me double check this using Python:

Assistant (python):
'nonclassifiable'.count('s')

Python output:
2

Assistant (text):
Python gives us 2.

My final answer is:

#### 2
```

## 附录 B：主要源码入口

- SFT 主程序：[`scripts/chat_sft.py`](../../scripts/chat_sft.py)
- 对话渲染和 loss mask：[`nanochat/tokenizer.py`](../../nanochat/tokenizer.py)
- GPT forward 和交叉熵：[`nanochat/gpt.py`](../../nanochat/gpt.py)
- Task mixture：[`tasks/common.py`](../../tasks/common.py)
- SmolTalk：[`tasks/smoltalk.py`](../../tasks/smoltalk.py)
- Identity JSONL：[`tasks/customjson.py`](../../tasks/customjson.py)
- MMLU：[`tasks/mmlu.py`](../../tasks/mmlu.py)
- GSM8K：[`tasks/gsm8k.py`](../../tasks/gsm8k.py)
- 拼写和字符计数：[`tasks/spellingbee.py`](../../tasks/spellingbee.py)
- 预训练 dataloader（用于对照）：[`nanochat/dataloader.py`](../../nanochat/dataloader.py)
