## Tokenizer之BPE算法介绍
 
### Overview
Tokenizer的目标是 "将字符序列映射为token编码" 。tokenizer训练完成后会得到一个
码表，作为字符序列到tokenId的映射，使用它可以完成字符序列与token序列之间的
encoding/decoding。

BPE是nanochat采用的tokenizer算法，核心就是一句话： 
```
反复把语料里最常见的相邻 token pair 合并成一个新 token。
```

BPE在初始化的时候只有256个基础token，用来编码一个byte，这样它天然能覆盖任意文本，
包括中文、emoji、乱码、代码。BPE的过程，就是根据token序列不断计算相邻token pair
的频率，并将最频繁出现的token pair合并产生新token，直到达到tokenizer vocabulary
size的上限。

**举个例子**

假设现在有一个 chunk："banana"， 初始按 byte/字符看成：
```
["b", "a", "n", "a", "n", "a"]
```
那么相邻 pair 是：
```
("b", "a")
("a", "n")
("n", "a")
("a", "n")
("n", "a")
```
统计频率：
```
("a", "n") 出现 2 次
("n", "a") 出现 2 次
("b", "a") 出现 1 次
```
假设先选 ("a", "n") 合并成新 token "an"：
```
["b", "an", "a", "n", "a"]
```
然后重新统计 pair：
```
("b", "an")
("an", "a")
("a", "n")
("n", "a")
```
再选最高频 pair 继续合并。


### BPE & Chunk
在`nanochat/rustbpe`的BPE实现中，并不是直接对整篇文档无限制合并，而是
先用 GPT 风格 regex 把文本切成 chunk。

**chunk 是什么？**

```nanochat```的chunk规则定义如下。
```
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?
[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

大致意思是按这些类型切：
```
英文缩写后缀: 's, 't, 're, 've, 'll ...
字母串: hello, world，也可能带一个前导标点/空格
数字: 1-2 位一组，nanochat 特意用 {1,2}
标点/符号串: !!!, (), :=, @#
换行
空白
```

例如：
```
"Hello, world! I'm 1234"
```
可能先切成类似：
```
["Hello", ",", " world", "!", " I", "'m", " ", "12", "34"]
```
然后 BPE 只在每个 chunk 内部做 byte pair merge，通常不跨 chunk 合并。在计算频率的时候，
也是以chunk为单位进行计算（每个chunk内部相邻的token一起算频率，跨chunk的token不算相邻token）。

BPE 的频率统计是跨所有 chunk 汇总的，但 pair 本身只在单个 chunk 内产生，不跨 chunk 边界。

### 频率
频率决定“哪个 pair 最值得变成一个新 token”。 假设语料里 "tion" 特别常见。如果你能把：
```
"t" + "i" -> "ti"
"ti" + "o" -> "tio"
"tio" + "n" -> "tion"
```
学成 token，那么每次出现 "tion"，token 数都会减少。

BPE 每新增一个 token，都是在消耗有限词表容量。你当然希望把容量用在最常见的模式上，因为它能在
整个语料里节省最多 token。

**简化例子**

假如有如下语料：
```
lowolowerer
```
可以看到下面的 pair 经常出现，就会逐步被合并：
```
l + o -> lo
lo + w -> low
e + r -> er
```
下面的单词按照规则进行tokenizing：
```
lower -> ["low", "er"]
lowest -> ["low", "est"]
```
这就是 BPE 的好处：它不是死记所有单词，而是学习可复用的高频子词片段。


### 总结
BPE 是一种很朴素的压缩策略：把最常见的相邻模式变短。LLM tokenizer 借用
了这个压缩思想，把文本压成更短、更稳定、还能无损还原的 token id 序列
