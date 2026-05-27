# nanochat 训练流程总览

本文面向第一次读 nanochat 的人，目标是先建立“一个 LLM 从数据到可聊天模型”的完整地图，再知道每个阶段应该从哪段入口代码开始看。

## 1. 一句话理解这个项目

nanochat 是一个极简的端到端 LLM 训练实验框架。它不只是训练 base model，而是覆盖了：

1. 下载预训练数据
2. 训练 tokenizer
3. 预训练 base model
4. 评测 base model
5. 监督微调成 chat model
6. 评测 chat model
7. 可选 RL
8. CLI / Web UI 推理
9. 生成训练报告

最推荐先读的总入口是：

- `runs/speedrun.sh`：8xH100 上训练一个 GPT-2 级别 nanochat 的参考完整流程。
- `runs/runcpu.sh`：CPU / Apple Silicon 上跑通小规模教育版流程。

如果你的核心目标是理解 LLM 完整训练流程，不要一开始陷入模型细节。先把 `runs/speedrun.sh` 从上到下读一遍，它就是主线剧本。

## 2. 完整训练主线

默认的完整训练链路由 `runs/speedrun.sh` 串起来：

```text
环境准备
  -> 下载预训练数据
  -> 训练 tokenizer
  -> 评测 tokenizer
  -> 预训练 base model
  -> 评测 base model
  -> 下载 SFT identity 数据
  -> SFT 微调 chat model
  -> 评测 chat model
  -> CLI / Web UI 聊天
  -> 生成 report
```

每个阶段的产物默认写到 `NANOCHAT_BASE_DIR`。如果没有显式设置，代码会使用 `~/.cache/nanochat`；`speedrun.sh` 会设置：

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
```

关键产物目录如下：

- 预训练数据：`$NANOCHAT_BASE_DIR/base_data_climbmix`
- tokenizer：`$NANOCHAT_BASE_DIR/tokenizer`
- base checkpoints：`$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>`
- SFT checkpoints：`$NANOCHAT_BASE_DIR/chatsft_checkpoints/<model_tag>`
- RL checkpoints：`$NANOCHAT_BASE_DIR/chatrl_checkpoints/<model_tag>`
- base eval CSV：`$NANOCHAT_BASE_DIR/base_eval`
- report：`$NANOCHAT_BASE_DIR/report`

## 3. 阶段 0：环境和报告初始化

入口：

- `runs/speedrun.sh`
- `nanochat/common.py`
- `nanochat/report.py`

`speedrun.sh` 先做几件事：

- 设置 `OMP_NUM_THREADS=1`
- 设置 `NANOCHAT_BASE_DIR`
- 使用 `uv sync --extra gpu` 安装依赖
- 如果没有设置 `WANDB_RUN`，就用 `dummy` 禁用 wandb
- 调用 `python -m nanochat.report reset` 初始化报告

你需要知道：

- 几乎所有中间文件都不写在项目目录，而是写到 `get_base_dir()` 返回的缓存目录。
- `nanochat/common.py` 负责设备选择、DDP 初始化、精度选择、日志等公共逻辑。
- 项目不用 `torch.amp.autocast`，而是用 `nanochat/common.py` 里的全局 `COMPUTE_DTYPE` 管理计算精度。

## 4. 阶段 1：下载预训练数据

入口：

- `python -m nanochat.dataset -n 8`
- `python -m nanochat.dataset -n 170`
- 代码：`nanochat/dataset.py`
- 数据重打包参考：`dev/repackage_data_reference.py`

流程：

1. `speedrun.sh` 先下载 8 个 shard，用于 tokenizer 训练。
2. 同时在后台下载更多 shard，给 base model 预训练使用。
3. 当前预训练数据是 Hugging Face 上的 `karpathy/climbmix-400b-shuffle` parquet shards。
4. `dataset.py` 总是额外下载最后一个 shard 作为 validation shard。

关键函数：

- `list_parquet_files()`：列出本地 parquet 文件。
- `parquets_iter_batched(split)`：按 train / val split 迭代 parquet row groups。
- `download_single_file(index)`：下载单个 shard。

新手要注意：

- 训练数据是纯文本 parquet，列名是 `text`。
- train split 使用除最后一个 shard 之外的文件，val split 使用最后一个 shard。
- 下载只是准备原始文本数据，真正变成 token batch 是后面的 dataloader 做的。

## 5. 阶段 2：训练 tokenizer

入口：

- 命令：`python -m scripts.tok_train`
- 代码：`scripts/tok_train.py`
- tokenizer 实现：`nanochat/tokenizer.py`

流程：

1. `tok_train.py` 通过 `parquets_iter_batched(split="train")` 读取预训练文本。
2. 每篇文档最多截取 `--doc-cap` 个字符，默认 10,000。
3. 最多读取 `--max-chars` 个字符，默认 2B。
4. 使用 `RustBPETokenizer.train_from_iterator(...)` 训练 BPE tokenizer。
5. 默认 vocab size 是 32,768。
6. 保存 tokenizer 到 `$NANOCHAT_BASE_DIR/tokenizer`。
7. 额外保存 `token_bytes.pt`，供 bits-per-byte 评测使用。

关键文件和函数：

- `scripts/tok_train.py`：训练入口。
- `RustBPETokenizer.train_from_iterator()`：训练 BPE 并构造 tiktoken encoding。
- `SPECIAL_TOKENS`：chat / tool use 所需的特殊 token。
- `get_tokenizer()`：后续所有训练和推理统一从缓存目录加载 tokenizer。
- `get_token_bytes()`：加载每个 token 对应的 byte 长度。

特殊 token 很重要：

```text
<|bos|>
<|user_start|>
<|user_end|>
<|assistant_start|>
<|assistant_end|>
<|python_start|>
<|python_end|>
<|output_start|>
<|output_end|>
```

这些 token 决定了 SFT 时如何把多轮对话渲染成 token 序列，也决定推理时如何识别用户、助手、工具调用和工具输出。

## 6. 阶段 3：评测 tokenizer

入口：

- 命令：`python -m scripts.tok_eval`
- 代码：`scripts/tok_eval.py`

它会把 nanochat tokenizer 和 GPT-2 / GPT-4 tokenizer 做压缩率对比。指标核心是：

```text
bytes per token = 文本 UTF-8 字节数 / token 数
```

你需要知道：

- tokenizer 好坏会影响训练效率和评测可比性。
- base model 训练中的 `val_bpb` 不是普通 loss，而是 bits per byte，用来减少不同 vocab size 带来的指标偏差。

## 7. 阶段 4：预训练 base model

入口：

- 命令：`torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 ...`
- 代码：`scripts/base_train.py`
- 模型：`nanochat/gpt.py`
- dataloader：`nanochat/dataloader.py`
- optimizer：`nanochat/optim.py`
- checkpoint：`nanochat/checkpoint_manager.py`

这是整个项目最核心、最耗算力的阶段。

### 7.1 base_train 做什么

`scripts/base_train.py` 大致流程：

1. 初始化设备和 DDP：`compute_init(...)`
2. 加载 tokenizer：`get_tokenizer()`
3. 根据 `--depth` 自动推导模型宽度、head 数等结构参数
4. 构造 `GPTConfig`
5. 创建 `GPT` 模型并初始化权重
6. 可选启用 FP8
7. `torch.compile(model)`
8. 根据 scaling laws 自动推导训练 token 数、batch size、学习率缩放、weight decay 缩放
9. 初始化 optimizer
10. 初始化预训练 dataloader
11. 进入训练循环
12. 周期性评测 `val_bpb`
13. 周期性评测 CORE metric
14. 周期性采样
15. 保存 checkpoint

### 7.2 depth 是项目的核心复杂度旋钮

项目最重要的设计是：用户主要只设置 `--depth`。

在 `base_train.py` 中：

```python
base_dim = depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
```

也就是说：

- `depth` 决定 transformer 层数。
- `aspect_ratio` 默认 64，用于从 depth 推导 hidden size。
- `head_dim` 默认 128，用于推导 attention head 数。
- 训练 token 数、batch size、学习率、weight decay 也会围绕模型规模自动推导。

新手读这里时要建立一个认识：nanochat 不是通用大配置框架，它故意把复杂度集中到少数几个旋钮上。

### 7.3 模型结构在哪里

入口：

- `nanochat/gpt.py`

核心类：

- `GPTConfig`
- `GPT`
- `Block`
- `CausalSelfAttention`
- `MLP`

模型特点：

- decoder-only causal Transformer
- rotary embeddings
- QK norm
- RMSNorm，无可学习 norm 参数
- untied token embedding 和 lm_head
- ReLU squared MLP
- 可选 sliding window attention pattern
- Flash Attention 3 优先，其他硬件回退到 PyTorch SDPA
- 支持 KV cache 推理
- optimizer 参数分组直接定义在 `GPT.setup_optimizer()`

训练 forward：

```text
token ids
  -> token embedding
  -> RMSNorm
  -> 多层 Transformer blocks
  -> RMSNorm
  -> lm_head
  -> logits
  -> cross entropy loss
```

### 7.4 预训练 dataloader 在哪里

入口：

- `nanochat/dataloader.py`

核心函数：

- `tokenizing_distributed_data_loader_with_state_bos_bestfit(...)`
- `tokenizing_distributed_data_loader_bos_bestfit(...)`

它做的事情：

1. 从 parquet 读取文本。
2. DDP 下不同 rank 读取不同 row group。
3. 用 tokenizer 把文本转 token。
4. 每个训练 row 都以 BOS 开头。
5. 用 best-fit packing 尽量把文档塞满固定长度序列。
6. 输出 `inputs` 和 `targets`，本质是 next-token prediction：

```text
inputs  = row[:, :-1]
targets = row[:, 1:]
```

预训练阶段没有对话 mask。所有 target token 都参与 next-token prediction。

### 7.5 预训练 checkpoint

入口：

- `nanochat/checkpoint_manager.py`

保存目录：

```text
$NANOCHAT_BASE_DIR/base_checkpoints/d<depth>/
```

文件形态：

```text
model_000000.pt
meta_000000.json
optim_000000_rank0.pt
optim_000000_rank1.pt
...
```

`meta_*.json` 很重要，里面包含：

- step
- val_bpb
- model_config
- user_config
- batch size / seq len
- dataloader resume state
- training loop state

后续 SFT 会从 base checkpoint 加载模型。

## 8. 阶段 5：评测 base model

入口：

- 命令：`torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16`
- 代码：`scripts/base_eval.py`
- CORE 评测底层：`nanochat/core_eval.py`
- BPB 评测底层：`nanochat/loss_eval.py`

`base_eval.py` 支持三类评测：

```text
--eval core,bpb,sample
```

默认三种都跑：

- `core`：DCLM CORE 风格的 in-context learning 任务。
- `bpb`：train / val bits per byte。
- `sample`：用一些简单 prompt 看 base model 补全文本能力。

关键函数：

- `evaluate_core(...)`
- `evaluate_bpb(...)`
- `Engine.generate_batch(...)`

你需要知道：

- base model 还不是 chat model，它只学了续写文本。
- base 评测关注的是“语言建模能力”和 ICL 能力，不是多轮对话体验。
- `speedrun.sh` 里用 CORE metric 衡量是否达到 GPT-2 级别。

## 9. 阶段 6：SFT，把 base model 变成 chat model

入口：

- 命令：`torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16`
- 代码：`scripts/chat_sft.py`
- 对话渲染：`nanochat/tokenizer.py`
- 任务数据：`tasks/*.py`

SFT 的目标是让模型学会：

- 按 chat 格式回复
- 理解 user / assistant special tokens
- 做多选题
- 做 GSM8K 数学题和简单 tool use
- 拼写、数数等特定能力
- 具备 nanochat 的 identity / personality

### 9.1 SFT 从哪里加载模型

`chat_sft.py` 先调用：

```python
model, tokenizer, meta = load_model("base", device, phase="train", ...)
```

也就是从 `$NANOCHAT_BASE_DIR/base_checkpoints` 加载 base model。

默认还会继承预训练 checkpoint 里的这些设置：

- `max_seq_len`
- `device_batch_size`
- `total_batch_size`
- `embedding_lr`
- `unembedding_lr`
- `matrix_lr`

### 9.2 SFT 数据混合

入口：

- `scripts/chat_sft.py`
- `tasks/smoltalk.py`
- `tasks/mmlu.py`
- `tasks/gsm8k.py`
- `tasks/customjson.py`
- `tasks/spellingbee.py`

训练 mixture：

```text
SmolTalk(train)
identity_conversations.jsonl
identity_conversations.jsonl
MMLU auxiliary_train x mmlu_epochs
GSM8K train x gsm8k_epochs
SimpleSpelling
SpellingBee
```

validation mixture：

```text
SmolTalk(test)
MMLU(test 子集)
GSM8K(test 子集)
```

`identity_conversations.jsonl` 在 `speedrun.sh` 中通过 curl 下载到 `$NANOCHAT_BASE_DIR`，它用于给 nanochat 注入身份和风格。

### 9.3 SFT 的 loss mask 是核心

SFT 不应该训练模型去预测用户说的话。它只应该训练 assistant 回复。

关键代码在 `RustBPETokenizer.render_conversation(...)`：

- user token 的 mask 是 0
- assistant 内容 token 的 mask 是 1
- tool output 的 mask 是 0
- padding 的 target 会被置为 `-1`

最终 `chat_sft.py` 构造：

```text
inputs:  token 序列左移前
targets: token 序列左移后
targets[mask == 0] = -1
```

模型 loss 使用 `ignore_index=-1`，所以只有 assistant 应该生成的 token 参与训练。

这是理解 SFT 和 pretraining 最大差别的关键。

### 9.4 SFT checkpoint

保存目录：

```text
$NANOCHAT_BASE_DIR/chatsft_checkpoints/d<depth>/
```

后续 CLI / Web / chat_eval 默认加载 SFT 模型。

## 10. 阶段 7：评测 chat model

入口：

- 命令：`torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft`
- 代码：`scripts/chat_eval.py`

支持任务：

```text
ARC-Easy
ARC-Challenge
MMLU
GSM8K
HumanEval
SpellingBee
```

两种评测方式：

- categorical eval：多选题，不采样，直接看候选答案 token 的 logits。
- generative eval：生成答案，再用任务自己的 `evaluate(...)` 判断对错。

关键函数：

- `run_chat_eval(...)`
- `run_categorical_eval(...)`
- `run_generative_eval(...)`

`chat_sft.py` 训练过程中也会周期性调用 `run_chat_eval(...)` 算 ChatCORE。

## 11. 阶段 8：可选 RL

入口：

- 命令：`python -m scripts.chat_rl`
- 分布式命令：`torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default`
- 代码：`scripts/chat_rl.py`

注意：RL 不在默认 `runs/speedrun.sh` 主线里，它是一个可选后训练阶段。

当前 RL 主要针对 GSM8K，用类似简化版 GRPO / REINFORCE 的方式：

1. 从 SFT checkpoint 加载模型。
2. 对每个 GSM8K 问题采样多个答案。
3. 用 `GSM8K.reward(...)` 给每个答案打 reward。
4. 用 `reward - mean(reward)` 做 advantage。
5. 对被采样出来的 assistant token 做 policy gradient。
6. 周期性评测 pass@k。
7. 保存到 `$NANOCHAT_BASE_DIR/chatrl_checkpoints`。

和 SFT 的关键差别：

- SFT 学的是数据集中已有的标准答案。
- RL 学的是模型自己采样出的答案，靠 reward 判断好坏。
- 当前实现没有 PPO ratio / clip，也没有 KL reference model。

## 12. 阶段 9：推理，CLI 和 Web UI

CLI 入口：

- 命令：`python -m scripts.chat_cli -p "Why is the sky blue?"`
- 代码：`scripts/chat_cli.py`

Web 入口：

- 命令：`python -m scripts.chat_web`
- 代码：`scripts/chat_web.py`
- 前端：`nanochat/ui.html`

推理核心：

- `nanochat/engine.py`

`Engine` 做的事情：

- 使用 KV cache 加速 autoregressive generation。
- 支持 batch generation。
- 支持 streaming generation。
- 处理 `<|python_start|>` / `<|python_end|>` 工具调用状态机。
- 当模型生成 python 表达式时，尝试用受限的 `eval` 计算，再把输出 token 强制插回生成流。

CLI 和 Web 都默认加载：

```text
source = sft
```

也就是 `$NANOCHAT_BASE_DIR/chatsft_checkpoints` 下的模型。可以用 `-i rl` 改为加载 RL checkpoint。

## 13. 训练流程总表

| 阶段 | 目的 | 命令入口 | 主要代码 | 主要产物 |
|---|---|---|---|---|
| 环境 / report | 安装依赖，初始化报告 | `runs/speedrun.sh` | `nanochat/common.py`, `nanochat/report.py` | `$NANOCHAT_BASE_DIR/report` |
| 数据下载 | 下载 parquet 文本 shard | `python -m nanochat.dataset -n ...` | `nanochat/dataset.py` | `base_data_climbmix/` |
| tokenizer 训练 | 训练 BPE tokenizer | `python -m scripts.tok_train` | `scripts/tok_train.py`, `nanochat/tokenizer.py` | `tokenizer/` |
| tokenizer 评测 | 看压缩率 | `python -m scripts.tok_eval` | `scripts/tok_eval.py` | report section |
| base 预训练 | next-token prediction | `torchrun ... -m scripts.base_train` | `scripts/base_train.py`, `nanochat/gpt.py`, `nanochat/dataloader.py` | `base_checkpoints/` |
| base 评测 | CORE / BPB / sample | `torchrun ... -m scripts.base_eval` | `scripts/base_eval.py`, `nanochat/core_eval.py`, `nanochat/loss_eval.py` | `base_eval/`, report section |
| SFT | 训练 chat 格式和任务能力 | `torchrun ... -m scripts.chat_sft` | `scripts/chat_sft.py`, `tasks/*.py`, `nanochat/tokenizer.py` | `chatsft_checkpoints/` |
| chat 评测 | ChatCORE / 任务准确率 | `torchrun ... -m scripts.chat_eval -- -i sft` | `scripts/chat_eval.py` | report section |
| RL 可选 | GSM8K reward 优化 | `torchrun ... -m scripts.chat_rl` | `scripts/chat_rl.py` | `chatrl_checkpoints/` |
| 推理 | CLI / Web 聊天 | `python -m scripts.chat_cli`, `python -m scripts.chat_web` | `nanochat/engine.py`, `scripts/chat_*.py` | 可交互服务 |

## 14. 新手必须知道的几个概念

### 14.1 Pretraining 和 SFT 的本质区别

预训练：

- 数据是普通文本。
- 目标是预测下一个 token。
- 所有 target token 都参与 loss。
- 学到的是语言建模能力、知识、基础推理模式。

SFT：

- 数据是 conversation。
- 输入中包含 user / assistant special tokens。
- 只对 assistant 应该生成的 token 计算 loss。
- 学到的是“如何按聊天格式回答”和特定任务行为。

### 14.2 base model 不能直接等价于 chatbot

base model 是文本续写器。给它一句话，它会继续补文本，但它未必遵守聊天格式，也未必知道该扮演 assistant。

chat model 是 SFT 之后的模型。它学会了：

```text
<|user_start|> 用户内容 <|user_end|>
<|assistant_start|> 助手内容 <|assistant_end|>
```

所以你在 CLI / Web 里对话时，代码会把消息转成这种 token 格式。

### 14.3 `--depth` 是理解 nanochat 配置的主线

nanochat 的设计理念是用一个主要旋钮控制模型规模：

- 小实验：`depth=4/6/12`
- GPT-2 级别 speedrun：当前参考脚本使用 `depth=24`

不要一开始到处找 YAML 配置。这个项目故意没有复杂配置系统。

### 14.4 DDP 和 batch size 的关系

训练脚本里的几个 batch 概念：

- `device_batch_size`：单张 GPU 一次 forward/backward 的 batch size。
- `max_seq_len`：每条序列长度。
- `ddp_world_size`：GPU / rank 数。
- `total_batch_size`：一次 optimizer step 覆盖的总 token 数。
- `grad_accum_steps`：为了达到 total batch size，需要累积多少个 micro-step。

公式在 `base_train.py` 和 `chat_sft.py` 中都有：

```python
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
```

### 14.5 BPB 是重要指标

BPB = bits per byte。

普通 cross entropy loss 会受 tokenizer vocab size 影响。BPB 用 token 对应的 byte 数归一化，更适合比较不同 tokenizer / 数据设置下的语言建模质量。

相关代码：

- `scripts/tok_train.py`：生成 `token_bytes.pt`
- `nanochat/loss_eval.py`：计算 BPB
- `scripts/base_train.py`：训练中周期性评测 `val_bpb`
- `scripts/base_eval.py`：评测 train / val BPB

### 14.6 CORE 和 ChatCORE 不一样

CORE：

- 用于 base model。
- 入口在 `scripts/base_eval.py`。
- 关注 in-context learning 风格的基础能力。

ChatCORE：

- 用于 chat model。
- 入口在 `scripts/chat_eval.py`，训练中也由 `chat_sft.py` 调用。
- 关注 SFT 后模型在 ARC、MMLU、GSM8K、HumanEval、SpellingBee 等任务上的表现。

### 14.7 checkpoint 加载默认会猜模型

`load_model(...)` 如果不传 `model_tag`，会在对应 checkpoint 目录里猜一个模型：

1. 优先找名字类似 `d<number>` 的最大 depth。
2. 否则找最近修改的目录。

所以当你本地有多个实验时，建议显式传：

```bash
--model-tag d24
```

否则你可能加载到自己没预期的 checkpoint。

## 15. 建议阅读顺序

第一轮，只看主线：

1. `runs/speedrun.sh`
2. `scripts/tok_train.py`
3. `scripts/base_train.py`
4. `scripts/chat_sft.py`
5. `scripts/chat_cli.py`

第二轮，补齐关键模块：

1. `nanochat/dataset.py`
2. `nanochat/dataloader.py`
3. `nanochat/tokenizer.py`
4. `nanochat/gpt.py`
5. `nanochat/checkpoint_manager.py`
6. `nanochat/engine.py`

第三轮，再看评测和任务：

1. `scripts/base_eval.py`
2. `scripts/chat_eval.py`
3. `tasks/common.py`
4. `tasks/gsm8k.py`
5. `tasks/mmlu.py`
6. `tasks/arc.py`
7. `tasks/smoltalk.py`

最后再看可选研究脚本：

- `runs/miniseries.sh`
- `runs/scaling_laws.sh`
- `scripts/chat_rl.py`
- `dev/LOG.md`
- `dev/LEADERBOARD.md`

## 16. 如果你要动手跑

有 GPU 并想按主线跑：

```bash
bash runs/speedrun.sh
```

只想在本地理解流程：

```bash
bash runs/runcpu.sh
```

更推荐新手先手动分阶段跑 `runs/runcpu.sh` 里的命令，而不是一次性跑完整脚本。这样你能清楚看到每个阶段产生了哪些文件。

## 17. 读代码时的心智模型

可以把 nanochat 想成四层：

1. 数据层：`nanochat/dataset.py`, `nanochat/dataloader.py`, `tasks/*.py`
2. 表示层：`nanochat/tokenizer.py`
3. 模型训练层：`nanochat/gpt.py`, `scripts/base_train.py`, `scripts/chat_sft.py`, `scripts/chat_rl.py`
4. 评测和产品层：`scripts/base_eval.py`, `scripts/chat_eval.py`, `nanochat/engine.py`, `scripts/chat_cli.py`, `scripts/chat_web.py`

完整 LLM 训练不是单个 `train.py`，而是一条产物依赖链：

```text
parquet 文本
  -> tokenizer
  -> token batch
  -> base checkpoint
  -> SFT checkpoint
  -> 可选 RL checkpoint
  -> chat inference
```

理解这条链，再去看每个脚本内部细节，会比从模型层开始硬啃容易得多。
