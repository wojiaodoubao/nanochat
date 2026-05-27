# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat 是一个用于训练大语言模型的极简实验框架。它面向单个 GPU 节点运行，代码量小、易修改，并覆盖了 LLM 的主要阶段：分词器训练、预训练、微调、评测、推理，以及聊天 Web UI。例如，你可以只花约 48 美元（约 2 小时 8XH100 GPU 节点）训练一个具备 GPT-2 能力的 LLM；而 GPT-2 在 2019 年的训练成本约为 43,000 美元。使用竞价实例时，总成本可能接近 15 美元。

更一般地说，nanochat 开箱即用地支持训练一组计算最优的小模型。你只需要设置一个复杂度旋钮：`--depth`，也就是 GPT Transformer 的层数（具备 GPT-2 能力的模型大约对应 depth 26）。其他超参数，例如 Transformer 宽度、注意力头数、学习率调整、训练步数、权重衰减等，都会自动以较优方式计算出来。

关于本仓库的问题，推荐使用 Devin/Cognition 的 [DeepWiki](https://deepwiki.com/karpathy/nanochat) 提问，也可以使用 GitHub 的 [Discussions](https://github.com/karpathy/nanochat/discussions)，或加入 Discord 上的 [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) 频道。

## Time-to-GPT-2 排行榜

当前开发重点是调优预训练阶段，因为它消耗最多计算量。受 modded-nanogpt 仓库启发，为了激励进展和社区协作，nanochat 维护了一个 “GPT-2 speedrun” 排行榜：衡量把 nanochat 模型训练到 GPT-2 级别能力所需的真实墙钟时间，能力由 DCLM CORE 分数衡量。[runs/speedrun.sh](runs/speedrun.sh) 脚本始终代表训练 GPT-2 级别模型并与其聊天的参考流程。当前排行榜如下：

| # | 时间 | val_bpb | CORE | 说明 | 日期 | Commit | 贡献者 |
|---|------|---------|------|------|------|--------|--------|
| 0 | 168 小时 | - | 0.2565 | 原始 OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline，略微过训练 | 2026-01-29 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 略微欠训练 **+fp8** | 2026-02-02 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | 将总 batch size 提升到 1M tokens | 2026-02-05 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | 数据集切换到 NVIDIA ClimbMix | 2026-03-04 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | 2026-03-09 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | 2026-03-14 | a825e63 | @karpathy |

最重要的指标是 “time to GPT-2”：在 8XH100 GPU 节点上超过 GPT-2（1.6B）CORE 指标所需的墙钟时间。GPT-2 的 CORE 分数为 0.256525。2019 年训练 GPT-2 约需 43,000 美元；而由于过去 7 年整个技术栈的进步，现在可以在更短时间内、以远低于 100 美元的成本完成类似能力训练。例如按当前约 3 美元/GPU/小时计算，一个 8XH100 节点约 24 美元/小时，2 小时约 48 美元。

关于如何解读和参与排行榜，请参见 [dev/LEADERBOARD.md](dev/LEADERBOARD.md)。

## 快速开始

### 环境配置

nanochat 使用 [uv](https://docs.astral.sh/uv/) 管理依赖。安装方式如下：

```bash
uv sync --extra gpu    # 用于 CUDA（A100/H100 等）
uv sync --extra cpu    # 或用于仅 CPU / MPS
source .venv/bin/activate
```

开发环境会额外安装 pytest、matplotlib、ipykernel、transformers 等：

```bash
uv sync --extra gpu --group dev
```

### 复现 GPT-2 并与其聊天

最有意思的玩法是训练自己的 GPT-2 并与它聊天。完整流程都在单个文件 [runs/speedrun.sh](runs/speedrun.sh) 中，该脚本设计为在 8XH100 GPU 节点上运行。你可以从喜欢的云厂商启动一台新的 8XH100 GPU 机器（例如作者使用并推荐 [Lambda](https://lambda.ai/service/gpu-cloud)），然后启动训练脚本：

```bash
bash runs/speedrun.sh
```

这个流程大约需要 3 小时，建议放在 screen 会话中运行。完成后，你可以通过类似 ChatGPT 的 Web UI 与模型对话。再次确认本地 uv 虚拟环境已经激活（运行 `source .venv/bin/activate`），然后启动服务：

```bash
python -m scripts.chat_web
```

随后访问终端中显示的 URL。请注意使用正确地址，例如在 Lambda 上需要使用所在节点的公网 IP 加端口，类似 [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/)。然后就可以像平常使用 ChatGPT 一样与自己的 LLM 对话：让它写故事或诗，问它你是谁以观察幻觉，问它为什么天空是蓝色，也可以问为什么是绿色。speedrun 得到的是一个约 4e19 FLOPs 能力级别的模型，体验上有点像和幼儿园小朋友聊天。

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

补充说明：

- 代码在 Ampere 8XA100 GPU 节点上也可以正常运行，只是会慢一些。
- 去掉 `torchrun` 后，所有代码也可以在单 GPU 上运行，并产生近似相同的结果（代码会自动切换到梯度累积），但需要等待约 8 倍时间。
- 如果你的 GPU 显存少于 80GB，需要调整部分超参数，否则可能 OOM / 显存不足。可以在脚本中查找 `--device-batch-size` 并降低它，例如从 32（默认）降到 16、8、4、2，甚至 1。再低的话就需要更深入了解训练流程并自行调整。
- 大部分代码是比较标准的 PyTorch，因此原则上可在任何支持 PyTorch 的设备上运行，例如 xpu、mps 等。但作者没有亲自覆盖所有这些路径，所以可能存在边缘问题。

## 研究

如果你是研究者并希望帮助改进 nanochat，两个值得关注的脚本是 [runs/scaling_laws.sh](runs/scaling_laws.sh) 和 [runs/miniseries.sh](runs/miniseries.sh)。相关文档可参考 [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420)。做快速实验时（约 5 分钟预训练），作者偏好的规模是训练 12 层模型（GPT-1 量级），例如：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

这会使用 wandb（run 名称为 `d12`），只在最后一步运行 CORE 指标，不采样也不保存中间 checkpoint。常见工作流是改一处代码，重新运行 d12（或 d16 等），观察它是否带来改进。判断一次运行是否有帮助时，可以关注 wandb 中的这些曲线：

1. `val_bpb`：验证损失，以不受词表大小影响的 bits per byte 为单位，作为 `step`、`total_training_time` 和 `total_training_flops` 的函数。
2. `core_metric`：DCLM CORE 分数。
3. VRAM 利用率、`train/mfu`（Model FLOPS utilization）、`train/tok_per_sec`（训练吞吐）。

示例见 [这里](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)。

需要注意的是，nanochat 的代码和配置围绕一个复杂度旋钮构建：Transformer 的 depth。这个整数会自动决定其他所有超参数，例如 Transformer 宽度、注意力头数、学习率调整、训练步数、权重衰减等，使训练出的模型接近计算最优。用户无需思考或设置这些参数，只是在通过 `--depth` 请求一个更小或更大的模型，其他部分会自动工作。通过扫描不同 depth，可以得到一组不同规模的、计算最优的 nanochat miniseries 模型。当前代码中，GPT-2 能力模型大约在 d24-d26 范围。任何候选改动都应足够有原则，能够适用于所有 depth 设置。

## 在 CPU / MPS 上运行

[runs/runcpu.sh](runs/runcpu.sh) 展示了在 CPU 或 Apple Silicon 上运行的极简示例。它会大幅缩小正在训练的 LLM，让训练能在几十分钟内完成。用这种方式不会得到很强的结果。

## 精度 / dtype

nanochat 不使用 `torch.amp.autocast`，而是通过单个全局 `COMPUTE_DTYPE` 显式管理精度，该变量定义在 [nanochat/common.py](nanochat/common.py)。默认值会根据硬件自动检测：

| 硬件 | 默认 dtype | 原因 |
|------|------------|------|
| CUDA SM 80+（A100、H100 等） | `bfloat16` | 原生 bf16 tensor cores |
| CUDA SM < 80（V100、T4 等） | `float32` | 无 bf16；可通过 `NANOCHAT_DTYPE=float16` 使用 fp16（会启用 GradScaler） |
| CPU / MPS | `float32` | 无低精度 tensor cores |

你可以通过 `NANOCHAT_DTYPE` 环境变量覆盖默认值：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # 强制 fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # 强制 bf16
```

工作原理：模型权重以 fp32 存储，以保持优化器精度；自定义 `Linear` 层会在前向传播时将权重转换为 `COMPUTE_DTYPE`。Embedding 会直接以 `COMPUTE_DTYPE` 存储以节省内存。这提供了与 autocast 类似的混合精度收益，同时保留对每个部分运行精度的显式控制。

注意：`float16` 训练会在 `base_train.py` 中自动启用 `GradScaler`，以防止梯度下溢。SFT 也支持该机制，但 RL 目前不支持。fp16 推理在各处都可以正常工作。

## 指南

作者发布了一些可能有用的指南，按从新到旧排列：

- [2026-02-01：Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420)：记录第一组 nanochat miniseries 模型。
- 为 nanochat 增加新能力，请参见 [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164)。
- 自定义你的 nanochat，请参见 Discussions 中的 [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139)，其中描述了如何通过合成数据生成以及将其混入 SFT 阶段来调整 nanochat 的性格。
- [2025-10-13：original nanochat post](https://github.com/karpathy/nanochat/discussions/1)：介绍 nanochat 的原始帖子，不过现在包含一些过时信息，模型也比当前 master 老很多、效果更差。

## 文件结构

```text
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # 生成 identity 合成数据示例
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # 预训练数据 shard 生成
├── nanochat
│   ├── __init__.py                 # 空文件
│   ├── checkpoint_manager.py       # 保存/加载模型 checkpoint
│   ├── common.py                   # 小型通用工具和便利函数
│   ├── core_eval.py                # 评测 base model 的 CORE 分数（DCLM 论文）
│   ├── dataloader.py               # 分词型分布式 DataLoader
│   ├── dataset.py                  # 预训练数据的下载/读取工具
│   ├── engine.py                   # 带 KV Cache 的高效模型推理
│   ├── execution.py                # 允许 LLM 将 Python 代码作为工具执行
│   ├── gpt.py                      # GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # 评测 bits per byte（而不是普通 loss）
│   ├── optim.py                    # AdamW + Muon 优化器，支持单 GPU 和分布式
│   ├── report.py                   # 编写 nanochat Report 的工具
│   ├── tokenizer.py                # GPT-4 风格的 BPE Tokenizer 封装
│   └── ui.html                     # nanochat 前端 HTML/CSS/JS
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries 训练脚本
│   ├── runcpu.sh                   # CPU/MPS 运行小示例
│   ├── scaling_laws.sh             # Scaling laws 实验
│   └── speedrun.sh                 # 训练约 100 美元级 nanochat d20
├── scripts
│   ├── base_eval.py                # Base model：CORE 分数、bits per byte、采样
│   ├── base_train.py               # Base model：训练
│   ├── chat_cli.py                 # Chat model：通过 CLI 对话
│   ├── chat_eval.py                # Chat model：评测任务
│   ├── chat_rl.py                  # Chat model：强化学习
│   ├── chat_sft.py                 # Chat model：SFT 训练
│   ├── chat_web.py                 # Chat model：通过 Web UI 对话
│   ├── tok_eval.py                 # Tokenizer：评估压缩率
│   └── tok_train.py                # Tokenizer：训练
├── tasks
│   ├── arc.py                      # 多选科学题
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 从任意 jsonl 对话创建 Task
│   ├── gsm8k.py                    # 8K 小学数学题
│   ├── humaneval.py                # 名称不完全准确；简单 Python 编程任务
│   ├── mmlu.py                     # 多领域多选题
│   ├── smoltalk.py                 # Hugging Face SmolTalk 数据集集合
│   └── spellingbee.py              # 教模型拼写/数字母的任务
├── tests
│   └── test_engine.py
└── uv.lock
```

## 贡献

nanochat 的目标是在低于 1000 美元预算内，推动可端到端使用的微型模型达到更好的水平。这里的可访问性既指整体成本，也指认知复杂度。nanochat 不是一个拥有海量配置项的 LLM “框架”；代码库里没有巨型配置对象、模型工厂或复杂的 if-then-else 结构。它是一个单一、内聚、极简、可读、易修改、非常适合 fork 的 “strong baseline” 代码库，设计目标是端到端运行并产出一个可以聊天的 ChatGPT 风格模型。目前最有意思的方向是缩短达到 GPT-2 的延迟，也就是让 CORE 分数超过 0.256525。目前大约需要 3 小时，但通过改进预训练阶段仍可继续提升。

当前 AI 政策：披露。提交 PR 时，请声明哪些部分有大量 LLM 参与，且这些部分不是你亲手写的，或你没有完全理解。

## 致谢

- nanochat 这个名字来自作者之前的项目 [nanoGPT](https://github.com/karpathy/nanoGPT)，后者只覆盖预训练。
- nanochat 也受 [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) 启发。该项目通过清晰指标和排行榜将 nanoGPT 游戏化，nanochat 借鉴了其许多想法，也在预训练部分借用了部分实现。
- 感谢 [HuggingFace](https://huggingface.co/) 提供 fineweb 和 smoltalk。
- 感谢 [Lambda](https://lambda.ai/service/gpu-cloud) 提供本项目开发使用的计算资源。
- 感谢首席 LLM whisperer Alec Radford 的建议和指导。
- 感谢仓库管理员 Sofie [@svlandeg](https://github.com/svlandeg) 对 nanochat issue、pull request 和 discussion 管理的帮助。

## 引用

如果 nanochat 对你的研究有帮助，请按如下方式引用：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 许可证

MIT
