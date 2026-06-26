# L002: softmax 与交叉熵

## 学习目标

理解分类任务中的 `logits -> softmax -> 概率 -> target -> CrossEntropyLoss`。

## 我的当前理解

分类模型输出的不是直接概率，而是一组原始分数 `logits`。logits 最大的位置，是模型当前最偏向的类别。

`softmax` 会把 logits 转成概率，所有类别概率加起来等于 1。

`target` 是训练数据提供的正确类别编号。`CrossEntropyLoss` 关注真实类别的概率是否足够高：真实类别概率越高，loss 越小；真实类别概率越低，loss 越大。

更准确地说，`CrossEntropyLoss` 看的是 target 类别对应的 logit 相对其他类别 logits 是否足够大。softmax 会把这种相对优势转换成概率，所以观察起来更直观。

和回归任务对比：

- 回归任务预测连续数值，用 `MSELoss` 衡量预测值和真实值之间的数值差距。
- 分类任务预测类别，用 `CrossEntropyLoss` 惩罚真实类别概率太低。

## 最小代码例子

参考：`src/dl_practice/softmax_cross_entropy.py`

## 实验结果观察

已运行 `python -m dl_practice.softmax_cross_entropy`。

观察到：

- target 类别的 logit 相对其他类别更高时，softmax 后 target 概率更高，loss 更低。
- target 类别的 logit 相对其他类别更低时，softmax 后 target 概率更低，loss 更高。
- target 类别的 logit 相对其他类别高很多时，softmax 后 target 概率接近 1，loss 很低。

## 我修改了什么

新增最小实验代码，用手写 logits 观察 softmax 概率和交叉熵 loss。

## 反例 / 边界条件

- logits 不是概率，可以是负数，也不要求加起来等于 1。
- `nn.CrossEntropyLoss` 的输入应是 logits，不需要先手动 softmax。
- target 通常是类别编号，例如 `target=1`，不是 one-hot 向量。

## 和当前工作项目的关系

生物序列语言模型也会输出 token logits。理解 logits、softmax 和交叉熵，有助于后续理解 DNA/蛋白语言模型如何预测下一个 token，以及如何基于 logits 做打分、生成和下游任务。

## 我已经验证过什么

- logits 是原始类别分数，不是概率。
- softmax 后的结果更直观，因为它把相对分数转换成概率。
- `CrossEntropyLoss` 和 `MSELoss` 的关注点不同：MSELoss 衡量数值距离，CrossEntropyLoss 关注真实类别概率是否足够高。
- 对分类任务来说，target 类别的相对优势越大，loss 越小。

## 还不确定的问题

- 为什么 PyTorch 的 `CrossEntropyLoss` 要直接接收 logits，而不是 softmax 后的概率。
- 多个样本组成 batch 时，logits 和 target 的 shape 应该如何对应。

## 下一步

- 理解为什么 PyTorch 的 `CrossEntropyLoss` 要直接接收 logits，而不是 softmax 后的概率。
- 尝试 batch size 大于 1 的 logits 和 target。
