# run_002_batch_and_logits_input

## 目标

理解 batch size 大于 1 时 logits 和 target 的 shape，并观察为什么 `CrossEntropyLoss` 应该直接接收 logits。

## 配置

- model: none
- loss: `CrossEntropyLoss`
- optimizer: none
- learning_rate: none
- epochs: none
- dataset: manual logits and targets

## 结果

- batch shape: `logits_shape=[2, 3]`，`target_shape=[2]`
- correct loss from logits: `0.1909332424402237`
- loss after manual softmax: `0.6706742644309998`

## 观察

- batch demo 中有 2 个样本、3 个类别，所以 logits 是一个 `2 x 3` 的矩阵；target 只有 2 个元素，因为每个样本只需要一个真实类别编号。
- `per_sample_loss=[0.1909332424402237, 0.18358828127384186]`，`mean_loss=0.18726076185703278`，说明默认 `CrossEntropyLoss` 会对 batch 内样本 loss 取平均。
- 直接把 logits 传给 `CrossEntropyLoss` 时，loss 是 `0.1909`；先手动 softmax 再传入时，loss 变成 `0.6707`。这说明 `CrossEntropyLoss` 期待输入是 logits，而不是概率。

## 结论

`CrossEntropyLoss` 的正确输入是 logits。它内部会用稳定的方式做 `log_softmax` 再计算 loss，所以训练时不需要也不应该先手动 softmax。

当 batch size 大于 1 时：

- `logits.shape == [batch_size, num_classes]`
- `target.shape == [batch_size]`

target 的每个元素对应 batch 中一个样本的真实类别编号。

## 下一步

- 用自己的话解释为什么 “softmax 更适合观察，logits 更适合训练输入”。
- 进入下一个最小分类模型例子，把手写 logits 过渡到真正的 `nn.Linear` 分类输出。
