## batch size 与并行训练语义

在数据并行训练里，常见几个 batch 概念：

| 名称 | 含义 |
|---|---|
| `device_batch_size` | 每个设备、每个 micro-step 处理的样本条数。 |
| `max_seq_len` | 每条样本序列的 token 长度。 |
| `ddp_world_size` | 分布式训练中的进程数，通常也就是 GPU 数。 |
| `total_batch_size` | 一次 optimizer step 对应的全局 batch token 数。注意这里单位是 token，不是样本条数。 |
| `grad_accum_steps` | 梯度累积步数。连续做多次 forward/backward 后，再执行一次 `optimizer.step()`。 |

在 `scripts/base_train.py` 中，关系是：

```python
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
```

也就是：

```text
每个 rank 每个 micro-step 的 token 数
= device_batch_size * max_seq_len

所有 rank 每个 micro-step 的 token 数
= device_batch_size * max_seq_len * ddp_world_size

梯度累积步数
= total_batch_size / 所有 rank 每个 micro-step 的 token 数
```

举例：

```text
device_batch_size = 16
max_seq_len = 2048
ddp_world_size = 8
total_batch_size = 524288 tokens
```

则：

```text
tokens_per_fwdbwd = 16 * 2048 = 32768
world_tokens_per_fwdbwd = 32768 * 8 = 262144
grad_accum_steps = 524288 / 262144 = 2
```

意思是：每张 GPU 每次处理 16 条长度为 2048 的序列，8 张 GPU 一次 micro-step 总共处理 262144 个 token。为了达到 524288 token 的全局 batch，需要累积 2 个 micro-step 的梯度，然后再更新一次参数。

如果想让单进程训练和多进程数据并行训练在优化意义上接近一致，核心是让它们每次 `optimizer.step()` 看到相同规模的 `total_batch_size`。

例如：

```text
单卡：
device_batch_size = 32
ddp_world_size = 1
grad_accum_steps = 8
max_seq_len = 2048

global tokens = 32 * 1 * 8 * 2048 = 524288
```

```text
8 卡：
device_batch_size = 32
ddp_world_size = 8
grad_accum_steps = 1
max_seq_len = 2048

global tokens = 32 * 8 * 1 * 2048 = 524288
```

这两种设置的全局 batch token 数相同。数学语义上，如果数据顺序、loss reduction、梯度同步方式、优化器状态都一致，那么一次参数更新应该近似等价于用一个更大的 batch 做单次训练。

但这只能说明训练语义一致，并不等价于 bit-level 可重放。

## bit-level 可重放是什么意思

`bit-level 可重放` 指的是：两次运行得到的数值结果在底层二进制表示上完全一致。

这里的 `bit` 指浮点数在计算机里的二进制位，不是 token，也不是数据集里的 bit。

例如两个 `float32` 数：

```python
a = torch.tensor(1.0000001192092896, dtype=torch.float32)
b = torch.tensor(1.0000002384185791, dtype=torch.float32)
```

它们十进制显示很接近，但底层 bit 不一样。bit-level 可重放要求两次训练中对应 tensor 的每个元素都完全相同。

所以要区分两个层次：

| 层次 | 含义 |
|---|---|
| 训练语义一致 | global batch、loss 缩放、优化器更新规则一致，loss 曲线通常接近。 |
| bit-level 可重放 | 每一步输入、随机数、kernel、通信顺序、浮点计算结果都完全一样。 |

`global batch size` 一致是可复现实验的重要条件，但不是 bit-level 可重放的充分条件。

## 为什么多进程训练不一定 bit-level 一致

即使 `total_batch_size` 相同，单进程和多进程训练也可能出现低位差异，常见原因包括：

| 来源 | 影响 |
|---|---|
| dataloader | shuffle 顺序、worker 随机种子、数据分片、resume 位置不同，会导致输入 batch 不同。 |
| 随机操作 | dropout、随机采样、初始化等依赖 RNG state。 |
| CUDA kernel | 并行 reduction 的合并顺序不同，浮点低位可能不同。 |
| matmul / attention / norm | Transformer 中大量矩阵乘、softmax、归约、归一化都会受计算顺序影响。 |
| mixed precision | FP16、BF16、FP8 对数值路径更敏感。 |
| Flash Attention / fused kernels | fused kernel 可能重排计算、分块计算，改变浮点合并路径。 |
| `torch.compile` | 可能融合或重排计算图。 |
| 分布式通信 | `all_reduce`、`reduce_scatter`、`all_gather` 本质也是浮点规约，通信算法会影响合并顺序。 |
| checkpoint resume | 如果没有保存 RNG state、dataloader state、optimizer state，就无法精确接续。 |

因此：

```text
total_batch_size 一致
=> 每次 optimizer step 的训练规模一致，优化语义接近

bit-level 可重放
=> 每一步输入、随机数、算子实现、通信顺序、浮点结果都一致
```

前者是并行训练中必须维护的训练语义；后者是更严格的工程目标，通常需要额外控制 seed、数据顺序、确定性算法、cuBLAS/NCCL 行为、checkpoint 状态等。

## 浮点数为什么会因为加法顺序产生低位差异

浮点数不是按“整数部分 + 小数部分”分别存储的，而是按二进制科学计数法存储：

```text
value = sign * significand * 2^exponent
```

以 `float32` 为例，它的结构是：

```text
sign:     1 bit
exponent: 8 bits
fraction: 23 bits
```

对普通规格化数来说，真实有效数字是 `1.fraction`，所以 `float32` 一共有 24 位二进制有效精度：1 位隐藏的整数位，加上 23 位 fraction。也就是说，浮点精度是相对于当前数值尺度的“有效数字精度”，不是固定保留多少位小数。

例如 `2^-24` 单独是可以被 `float32` 精确表示的：

```text
2^-24 = 1.0 * 2^-24
```

它只需要有效数字 `1.0` 和指数 `-24`。但 `1.0 + 2^-24` 要表示成 `1.x * 2^0` 的形式：

```text
1.0 + 2^-24
= 1.000000000000000000000001₂
```

最后那个 `1` 落在小数点后的第 24 位，而 `float32` 在 `1.x` 这个尺度下只能保存 23 位 fraction。这个精确结果放不下，就必须舍入回最近的 `float32` 数。

`float32` 在 `1.0` 附近相邻两个可表示数的间隔是：

```text
2^-23
```

所以 `2^-24` 正好是半个间隔。IEEE 754 默认的 round-to-nearest-even 规则不是“总是向上进位”，而是舍入到最近值；正好在中间时选尾数最低位为偶数的那个。因此：

```text
float32(1.0 + 2^-24) == 1.0
```

这也解释了为什么 reduction 的合并顺序会影响低位。设：

```text
a = 1.0
b = 2^-24
c = 2^-24
```

第一种合并顺序：

```text
(a + b) + c
= float32(float32(1.0 + 2^-24) + 2^-24)
= float32(1.0 + 2^-24)
= 1.0
```

第二种合并顺序：

```text
a + (b + c)
= float32(1.0 + float32(2^-24 + 2^-24))
= float32(1.0 + 2^-23)
= 1.0 + 2^-23
```

这两个结果的底层 bit 不同：

```text
(a + b) + c = 1.0                         = 0x3f800000
a + (b + c) = 1.00000011920928955078125   = 0x3f800001
```

它们只差了 1 个 `float32` 的最低有效位，但这已经不是 bit-level 一致了。并行 reduction、CUDA kernel 内部规约、NCCL `all_reduce` 等都可能改变加法树；数学上都是求和，浮点上却可能因为每一步都要舍入到有限有效位而得到不同低位。