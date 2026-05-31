## 常见问题

### 为什么需要tokenizer？
1. 把文本变成模型能处理的整数
2. 控制序列长度：tokenizer后，相比于一个byte一个token，能够实现一定的压缩。
3. 保证开放词表能力：既能压缩，又能表示任意token。
4. 提供特殊控制符：LLM 不只是处理普通文本，还要知道结构，这些特殊 token
   可以明确告诉模型：“这里是用户消息开始”、“这里是助手回答开始”、“这里是工具调用”。
   * <|bos|>
   * <|user_start|>
   * <|assistant_start|>
   * <|python_start|>

### Tokenizer和大模型Embedding的区别
Tokenizer是将bytes sequence翻译为tokenId sequence。Embedding是将
tokenId映射为向量。

**相同点**
* 它们都是在做一一映射。

**不同点**
* embedding是LLM模型的一部分，每个token映射到的向量初始也是随机产生的，
  随着模型训练（反向传播）不断更新。

**完整过程串联**

你可以把输入分成两层看：
* 原始输入文本：字符串
* tokenizer 输出：token id 整数
* embedding lookup 输出：向量

真正“输入给模型”的最初形式是 token id：
```
input_ids = [101, 35, 998, 17]
```
但 Transformer 不能直接处理整数 id，所以模型内部第一层通常是 embedding 表：
```
nn.Embedding(vocab_size, n_embd)
```
它是一个可训练参数矩阵。

查表过程是：
```
token id 101 -> embedding_table[101] -> 一个 n_embd 维向量
```
这个向量是“当前 embedding 参数”里查出来的。训练开始时 embedding 表随机初始化，训练过程中通过
loss 反向传播更新。

所以 embedding 既可以被口语化地说成“输入向量”，也确实是模型内部的可训练参数。更准确地说：
* token id 是输入
* embedding 表是模型参数
* embedding 向量是由输入 id 查模型参数得到的激活值

nanochat 里这个表就是模型里的：
```
"wte": nn.Embedding(padded_vocab_size, config.n_embd)
```
它在 checkpoint 里保存，和 Transformer block、lm_head 一样都是模型权重。


### Tokenizer词表越大越好还是越小越好？（还需要进一步研究）
**大致的理解**

* 词表越小，embedding lookup表就小（模型参数少），但输入的token就多。
* 词表越大，embedding lookup表就大（模型参数多），但输入的token就少。
* 模型参数越多，训练时需要的显存就越大。
* 输入的token越多，也会影响lm head、激活显存什么的。

词表更大 -> 常见文本被压成更少 token -> 同一段原文对应的 T 更小

如果你固定“原文字符数”，那更大的词表可能让 token 数下降，于是 Transformer 主体处理的序列更
短，attention/MLP 的中间激活也可能更少。

**GPT分析**
* 词表变大，T 通常会变小，但不代表越大越好。

你说的方向有一部分是对的：
```
vocab_size 大 -> 文本 token 数可能减少 -> T 可能变小 -> Transformer 主体处理的序列变短
```
如果 B 和 n_embd 不变，[B, T, n_embd] 这部分激活确实会变小。

但词表变大同时会让另外两块变大：
```
embedding 表: vocab_size × n_embd
lm_head: n_embd × vocab_size
```
尤其是 lm_head。训练时每个位置都要预测下一个 token，输出 logits 是：
```
[B, T, vocab_size]
```
所以词表变大后，虽然 T 可能下降，但 vocab_size 上升了。输出层和 loss 的开销大致跟这个有关：
```
B × T × vocab_size
```
更完整一点，lm_head 矩阵乘法近似是：
```
B × T × n_embd × vocab_size
```
所以不是只看 [B, T, n_embd]，还要看：
```
[B, T, vocab_size]
```
举个简化例子：
```
方案 A:
vocab_size = 32k
T = 2048

方案 B:
vocab_size = 128k
T = 1700
```
B 的 T 少了约 17%，但 vocab 变成 4 倍。输出 logits 和 lm_head 相关开销可能反而大很多。

另外，词表越大还有几个问题：

- embedding 参数更多
- lm_head 参数更多
- optimizer state 更多，训练显存更大
- checkpoint 更大
- 长尾 token 更多，很多 token 训练次数少，表示学不好
- softmax 候选更多，推理采样成本更高

所以 tradeoff 是：

词表小：
token 序列长，Transformer 主体更贵

词表大：
token 序列短，但 embedding/lm_head/softmax 更贵

存在一个折中点。nanochat 默认 32768，很多 GPT 类小模型也常用 32k 左右，就是因为它在压缩率、
参数量、训练成本之间比较平衡。

再补一句：训练时很多框架并不会因为“这篇文本 token 少了”就自动减少 T。通常训练配置固定
max_seq_len=2048，dataloader 会把更多文本 pack 到 2048 token 里。于是词表变大更多体现为：

同样 2048 token 能装下更多原始文字

而不是每个 batch 一定变成更短的 T。
