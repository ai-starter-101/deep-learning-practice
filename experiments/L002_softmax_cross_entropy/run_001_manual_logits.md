# run_001_manual_logits

## 目标

用手写 logits 理解 softmax 概率和 `CrossEntropyLoss` 的关系。

## 配置

- model: none
- loss: `CrossEntropyLoss`
- optimizer: none
- learning_rate: none
- epochs: none
- dataset: manual logits and targets

## 结果

- case 1: target 类别 logit 最高，loss 低。
- case 2: target 类别 logit 最低，loss 高。
- case 3: target 类别 logit 很高，loss 很低。

## 观察

实验验证了：交叉熵不是看某个 logit 的绝对大小，而是看 target 类别 logit 相对其他类别 logits 是否足够大。

softmax 后的概率更适合观察这个关系：

- target probability 高，则 loss 低。
- target probability 低，则 loss 高。
- target probability 接近 1，则 loss 接近 0。

## 结论

`CrossEntropyLoss` 适合分类任务，因为它衡量的是模型给真实类别的概率是否足够高。它和 `MSELoss` 不同：`MSELoss` 比较预测数值和真实数值的距离；`CrossEntropyLoss` 比较模型给真实类别的相对信心。

## 下一步

- 尝试 batch size 大于 1 的 logits 和 target。
- 理解为什么 `nn.CrossEntropyLoss` 直接接收 logits，而不是接收手动 softmax 后的概率。
