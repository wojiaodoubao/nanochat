## 什么是BPB
在 LLM 训练里，bpb（bits per byte） 通常充当的是：验证集上的归一化评估指标，用来衡量模型
对原始文本的建模能力。

模型训练期间使用损失函数来评估模型，训练完成后可以使用bpb来衡量模型效果。在理解bpb
之前，需要先理解一下模型的损失函数。


### LLM训练时的损失函数
Transformer本质上是通过输入的token序列预测下一个token。对于训练集中任意一个token序列，
我们对其中的任意一个token，都可以根据其前序token序列结合model，求得当前token的概率，
记录为```p(token_i)```。
训练过程，就是对训练集中每一个序列的每一个token的概率计算均值，目标是使均值最小。

损失函数表示如下：
```
mean(-ln p(token_i)) 或 mean(-log p(token_i))
```
关于损失函数，有几个点要说明：
1. 损失函数中要求对p取自然对数，因为p是一个0~1之间的值，自然对数小于等于0，因此又乘了-1。
2. 损失函数采用 log(p) 也是常见技巧，另一个常见的是softmax，还有各种归一化，可以以后研究下。
3. log(n)在数学中表示常用对数，在机器学习里见到log通常是自然对数，在计算机其他领域如信息论
   里见到log通常是 以2为底n的对数。


### BPB的原理
BPB的作用之一是“衡量不同tokenizer的效果”，除此之外还有其他作用，但我们先以此为切入点进行理解。

在model训练完成后，我们需要在验证集上验证模型效果。使用token loss是不太可行的，因为不同model
的tokenizer可能差异很大，例如有的size是256，有的size是65532，计算出来的token loss不能直接
进行比较。此时就可以使用BPB。

BPB的计算公式如下：
```
bpb = total_nats / (math.log(2) * total_bytes)
```
含义是：
```
bpb = 所有目标 token 的总 loss，换算成 bits / 这些 token 对应的总 byte 数
```
也就是对于测试集中的任意一个token序列，分母是```token序列转换为bytes后的总byte数```。分子是
```所有目标token的总loss换算成bits```。

分母部分比较好理解，因为测试集中的字符序列是固定的，分母本质上是将token序列还原为字符序列后，
字符序列的总byte数。其值不受tokenizer的影响。

分子部分使用了“信息论”里压缩编码的思想，它的想法是：语言模型越好，说明它越能准确预测下一个 token，
也就是给真实文本更高概率。在“信息论”中，一个事件概率越高，编码它需要的 bit 数越少。如果一个token
的概率为p，那么编码这个token需要：```-log2(p) bits```，可使得最后编码所有token使用的bit最少。

这有点类似“哈夫曼编码”，目标一致都是使编码后的bit最少，但更进一步因为它允许bits不是整数，
而哈夫曼编码中bits必须是整数。

于是得到分子部分的完整公式如下，一个事件概率越高，编码它需要的 bit 数越少，total_nats越小。
```
total_nats = sum(-log2 p_i)
```

bpb 越低，代表：
* 模型越能预测文本
* 理论上越能把文本压缩得更短
* 语言建模效果越好


### BPB的应用
BPB用来回答几个问题。

**1. 作为验证指标**

训练时模型优化的是 token loss：
```
mean(-ln p(token_i))
```
但评估时可以看：
```
  bpb = sum(-log2 p(token_i)) / 原始文本 byte 数
```
所以 bpb 越低，说明模型越能预测验证集文本。

> 在 nanochat 里就是这个角色：
> * 每隔 N step，在 validation set 上计算 val_bpb
> * 记录最低的 val_bpb
> * 用它判断模型有没有变好

**2. 用来比较不同 tokenizer**

普通 token loss 受 tokenizer 影响很大。

比如同一段文本：
```
Tokenizer A: 1000 tokens
Tokenizer B: 1500 tokens
```
如果直接比较平均 token loss，意义不太稳定，因为 token 粒度变了。
bpb 把结果归一化到原始 byte： “每个原始 byte 平均需要多少预测 bits”。

这样更适合比较：
* 不同 vocab size
* 不同 tokenizer
* 不同 tokenization 策略

**3. 从压缩角度评估语言建模能力**

语言建模可以理解成给文本分配概率：
```
P(text) = P(t1) * P(t2 | t1) * P(t3 | t1,t2) ...
```
模型给真实文本的概率越高：```sum(-log2 p_i)```越小。

所以 bpb 可以理解成：
```
如果用这个模型来理论压缩文本，每个 byte 平均要花多少 bit
```
越低表示越好。

**4. 用于模型选择和监控**

训练中通常会记录：
* train loss
* val loss
* val bpb
* learning rate
* grad norm

其中 val_bpb 可以用来：
* 判断是否过拟合
* 选择 checkpoint
* 比较训练配置
* 比较 tokenizer
* 比较模型规模

比如两个模型 token loss 不好直接比，但如果它们在同一个验证集上的 bpb 分别是：
```
model A: 1.05 bpb
model B: 0.98 bpb
```
通常可以说 B 的语言建模效果更好。

## 一句话总结
bpb 在 LLM 训练中通常是一个评估指标，不是核心训练目标。它把模型预测真实文本的总
损失换算成“每个原始 byte 的 bits 成本”，因此比普通 token loss 更适合跨
tokenizer、跨 vocab size 比较模型的语言建模能力。