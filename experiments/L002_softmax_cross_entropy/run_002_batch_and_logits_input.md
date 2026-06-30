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

- batch shape:
- correct loss from logits:
- loss after manual softmax:

## 观察

待补充。

## 结论

待补充。

## 下一步

- 运行 `python -m dl_practice.softmax_cross_entropy`。
- 记录 batch demo 的输出。
- 解释 `logits.shape == [batch_size, num_classes]` 和 `target.shape == [batch_size]`。
