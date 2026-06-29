## 预训练命令
在`speedrun.sh`中，完成数据集下载以及tokenizer训练后，下一步是模型预训练。
```bash
# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
```

这里需要关注一下torchrun命令。
```
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-
ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
```

可以拆成两层理解。

第一层是 torchrun 启动分布式进程：

- torchrun：PyTorch 的分布式启动器。
- --standalone：单机训练，不需要多机 rendezvous 配置。
- --nproc_per_node=8：在本机启动 8 个 Python 进程，通常对应 8 张 GPU。
- -m scripts.base_train：每个进程都执行等价于：
```
python -u -m scripts.base_train ...
```
- 中间单独的 --：分隔 torchrun 自己的参数和传给 scripts.base_train 的参数。

第二层是传给训练脚本的配置：

- --depth=24：训练 24 层 Transformer，也就是注释里说的 d24 model。
- --target-param-data-ratio=8：按“训练 token 数 / 参数量”比例为 8 来自动算训练步数。对应 scripts/
base_train.py:269 和 scripts/base_train.py:349。

- --device-batch-size=16：每张 GPU 每个 micro-batch 放 16 条序列。默认 max_seq_len=2048，所以每个
rank 每次 forward/backward 是 16 * 2048 个 token。

- --fp8：启用 FP8 训练，主要面向 H100+ CUDA GPU。
- --run=$WANDB_RUN：传 wandb run 名；如果是 dummy 就不用真实 wandb。

它启动训练的链路是：

1. torchrun 启动 8 个进程，并给每个进程设置 RANK、LOCAL_RANK、WORLD_SIZE 等环境变量。
2. 每个进程运行 scripts/base_train.py。
   * base_train.py进行参数解析。
   * 调 compute_init()：它会 根据 LOCAL_RANK 绑定 对应 GPU，并初始化 NCCL process group。
   * 脚本加载 tokenizer、建模型、初始化权重、启用 FP8、torch.compile、创建 optimizer 和
      dataloader。
   * 真正训练循环从 scripts/base_train.py:416 的 while True: 开始；核心训练步骤在 scripts/
base_train.py:510：forward、backward、取下一批数据、optimizer.step()、清梯度、打印日志。

**一个容易忽略的点**：这里不是通过 DistributedDataParallel(model) 包模型。项目里是自定义分布式
optimizer：如果检测到 torchrun 环境，nanochat/gpt.py:410 会选择 DistMuonAdamW，它在 nanochat/
optim.py:299 里用 reduce_scatter / all_reduce / all_gather 做梯度和参数同步。数据加载也会按 rank
分片，见 nanochat/dataloader.py:33 和 nanochat/dataloader.py:62。

### base_train的参数与含义 

| 参数 | 默认值 | 类别 | 含义 |
|---|---:|---|---|
| `--run` | `dummy` | 日志 | wandb run 名称。`dummy` 表示不启用真实 wandb 上传。 |
| `--device-type` | 空字符串 | 运行设备 | 指定训练设备：`cuda`、`cpu` 或 `mps`。空字符串表示自动检测。 |
| `--fp8` | 关闭 | FP8 | 启用 FP8 训练，主要用于 H100+ CUDA GPU。 |
| `--fp8-recipe` | `tensorwise` | FP8 | FP8 缩放策略，可选 `tensorwise` 或 `rowwise`。`tensorwise` 更快，`rowwise` 更精细但更慢。 |
| `--depth` | `20` | 模型结构 | Transformer 层数。`speedrun.sh` 中传 `--depth=24`，表示训练 24 层模型。 |
| `--aspect-ratio` | `64` | 模型结构 | 控制模型宽度，基础关系是 `model_dim = depth * aspect_ratio`，实际会向上取整到 `head_dim` 的倍数。 |
| `--head-dim` | `128` | 模型结构 | 每个 attention head 的维度。 |
| `--max-seq-len` | `2048` | 模型结构 | 最大上下文长度，也是每条训练序列的 token 长度。 |
| `--window-pattern` | `SSSL` | 模型结构 | 各层 attention 窗口模式。`L` 表示 full attention，`S` 表示 half-context sliding window，字符串会按层循环使用。 |
| `--num-iterations` | `-1` | 训练时长 | 直接指定优化步数。`-1` 表示不使用该方式。 |
| `--target-flops` | `-1.0` | 训练时长 | 按目标总 FLOPs 反推训练步数。`-1` 表示不使用该方式。 |
| `--target-param-data-ratio` | `12` | 训练时长 | 按“训练 token 数 / 参数量”反推训练 token 数和步数。与 `--num-iterations`、`--target-flops` 同时存在时优先级最低。 |
| `--device-batch-size` | `32` | 优化 | 每个设备每个 micro-step 的样本条数。如果显存不足，通常降到 `16`、`8`、`4` 等。 |
| `--total-batch-size` | `-1` | 优化 | 全局 batch 的 token 数，不是样本条数。`-1` 表示根据 scaling law 自动计算。 |
| `--embedding-lr` | `0.3` | 优化 | token embedding 参数的 AdamW 学习率。 |
| `--unembedding-lr` | `0.008` | 优化 | unembedding / lm head 参数的 AdamW 学习率。 |
| `--weight-decay` | `0.28` | 优化 | Muon 优化器中矩阵参数的 weight decay。 |
| `--matrix-lr` | `0.02` | 优化 | Transformer 矩阵参数的 Muon 学习率。 |
| `--scalar-lr` | `0.5` | 优化 | 标量参数的 AdamW 学习率，例如 `resid_lambdas`、`x0_lambdas` 等。 |
| `--warmup-steps` | `40` | 学习率调度 | 学习率 warmup 步数。 |
| `--warmdown-ratio` | `0.65` | 学习率调度 | 训练后多少比例用于线性降低学习率。默认后 65% 步数进入 warmdown。 |
| `--final-lr-frac` | `0.05` | 学习率调度 | 最终学习率占初始学习率的比例。默认降到 5%。 |
| `--resume-from-step` | `-1` | 恢复训练 | 从指定 step 的 checkpoint 恢复训练。`-1` 表示不恢复。 |
| `--eval-every` | `250` | 评估 | 每多少步评估一次 validation BPB。`-1` 表示关闭。 |
| `--eval-tokens` | `80*524288` | 评估 | 每次 validation loss 评估使用的 token 数。 |
| `--core-metric-every` | `2000` | 评估 | 每多少步评估一次 CORE metric。`-1` 表示关闭。 |
| `--core-metric-max-per-task` | `500` | 评估 | CORE metric 中每个任务最多评估的样本数。 |
| `--sample-every` | `2000` | 评估 | 每多少步从当前模型生成样例文本。`-1` 表示关闭。 |
| `--save-every` | `-1` | 保存 | 每多少步保存一次 checkpoint。`-1` 表示只在训练结束时保存。 |
| `--model-tag` | `None` | 输出 | 覆盖 checkpoint 输出目录名。默认使用 `d{depth}`，例如 `d24`。 |

## 上面那段话要逐行解读一下
1. base_train都有哪些参数可调整，分别什么含义？
1. compute_init做了什么？
2. 什么是NCCL？
3. 如何实现的DistMuonAdamW/不使用DDP而自己实现训练？torch只负责提供dataloader、提供算子？
