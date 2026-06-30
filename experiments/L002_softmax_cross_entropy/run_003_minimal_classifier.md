# run_003_minimal_classifier

## 目标

把手写 logits 过渡到一个真实的最小分类模型：`x -> nn.Linear -> logits -> CrossEntropyLoss -> backward -> optimizer.step`。

## 配置

- model: `nn.Linear(in_features=2, out_features=3)`
- loss: `CrossEntropyLoss`
- optimizer: `SGD`
- learning_rate: demo default
- epochs: `80`
- dataset: 6 个手写二维样本，3 个类别

## 结果

- forward demo: `x_shape=[6, 2]`，`y_shape=[6]`，`logits_shape=[6, 3]`
- initial loss: `1.3848`
- initial accuracy: `0.3333`
- one training step: `loss_before_step=1.3848`，`loss_after_step=1.2245`
- epoch 10: `loss=0.4146`，`accuracy=1.0`
- epoch 80: `final_loss=0.0511`，`final_accuracy=1.0`

## 观察

- 刚初始化时，模型基本接近随机猜测，3 分类准确率约 `0.3333`。
- 只做一次 `backward + optimizer.step`，loss 就从 `1.3848` 降到 `1.2245`，说明梯度方向有效。
- 多轮训练后，预测类别从混乱逐步对齐 target，最终 6 个样本全部分类正确。
- `weight_grad_shape=[3, 2]`，`bias_grad_shape=[3]`，对应 3 个类别和 2 个输入特征。

## 结论

L002 已经从“理解 logits 和交叉熵直觉”推进到“跑通最小分类训练闭环”。分类任务的完整链路已经验证：

`x -> model -> logits -> CrossEntropyLoss -> backward -> optimizer.step`

## 下一步

- 用自己的话解释为什么 softmax 更适合观察，logits 更适合训练输入。
- 继续理解 `log_softmax + NLLLoss` 为什么比手动 `softmax -> log` 更稳定。
