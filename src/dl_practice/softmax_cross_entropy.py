from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# 1. 手写 logits：理解 CrossEntropyLoss 吃什么
# ============================================================

def run_softmax_cross_entropy_demo() -> list[dict[str, object]]:
    """
    CrossEntropyLoss 的输入是 raw logits，不是 softmax 之后的概率。

    logits shape: [batch_size, num_classes]
    target shape: [batch_size]
    target value: 每个样本的正确类别编号，例如 0/1/2
    """
    loss_fn = nn.CrossEntropyLoss()
    cases = [
        {
            "name": "correct_class_has_highest_logit",
            "logits": torch.tensor([[0.2, 3.0, 1.1]]),
            "target": torch.tensor([1]),
        },
        {
            "name": "correct_class_has_lowest_logit",
            "logits": torch.tensor([[0.2, 3.0, 1.1]]),
            "target": torch.tensor([0]),
        },
        {
            "name": "more_confident_correct_class",
            "logits": torch.tensor([[0.2, 6.0, 1.1]]),
            "target": torch.tensor([1]),
        },
    ]

    results: list[dict[str, object]] = []
    for case in cases:
        logits = case["logits"]
        target = case["target"]
        probabilities = torch.softmax(logits, dim=1)
        loss = loss_fn(logits, target)
        predicted_class = torch.argmax(logits, dim=1)
        target_probability = probabilities[0, target.item()].item()

        results.append(
            {
                "name": case["name"],
                "logits": logits.tolist()[0],
                "probabilities": probabilities.tolist()[0],
                "target": target.item(),
                "predicted_class": predicted_class.item(),
                "target_probability": target_probability,
                "loss": loss.item(),
            }
        )
    return results


# ============================================================
# 2. batch shape：一个 batch 里有多个样本
# ============================================================

def run_batch_shape_demo() -> dict[str, object]:
    logits = torch.tensor(
        [
            [0.2, 3.0, 1.1],
            [2.5, 0.1, 0.3],
        ]
    )
    target = torch.tensor([1, 0])

    probabilities = torch.softmax(logits, dim=1)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, target)
    per_sample_loss = nn.CrossEntropyLoss(reduction="none")(logits, target)
    predicted_classes = torch.argmax(logits, dim=1)

    return {
        "logits_shape": list(logits.shape),
        "target_shape": list(target.shape),
        "logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "target": target.tolist(),
        "predicted_classes": predicted_classes.tolist(),
        "per_sample_loss": per_sample_loss.tolist(),
        "mean_loss": loss.item(),
    }


# ============================================================
# 3. 反例：不要先手动 softmax 再喂给 CrossEntropyLoss
# ============================================================

def run_logits_vs_probabilities_demo() -> dict[str, float]:
    logits = torch.tensor([[0.2, 3.0, 1.1]])
    target = torch.tensor([1])

    probabilities = torch.softmax(logits, dim=1)
    loss_fn = nn.CrossEntropyLoss()
    correct_loss = loss_fn(logits, target)
    loss_after_manual_softmax = loss_fn(probabilities, target)

    return {
        "correct_loss_from_logits": correct_loss.item(),
        "loss_after_manual_softmax": loss_after_manual_softmax.item(),
    }


# ============================================================
# 4. 下一步：把“手写 logits”换成真正的 nn.Linear 模型
# ============================================================

class TinyLinearClassifier(nn.Module):
    """
    最小分类模型：输入特征 x -> nn.Linear -> logits。

    这里没有手动写 softmax，因为训练时 CrossEntropyLoss 直接接收 logits。
    只有在“观察/解释预测结果”时，才额外用 softmax 把 logits 转成概率。
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return logits


def make_toy_classification_data() -> tuple[torch.Tensor, torch.Tensor]:
    """
    构造一个极小的 3 分类数据集。

    x shape: [6, 2]，6 个样本，每个样本 2 个特征。
    y shape: [6]，每个样本对应一个类别 id。

    类别含义可以理解为：
    class 0: 靠近左上
    class 1: 靠近右上
    class 2: 靠近下方
    """
    x = torch.tensor(
        [
            [-2.0, 2.0],
            [-1.5, 1.5],
            [2.0, 2.0],
            [1.5, 1.0],
            [0.0, -2.0],
            [0.5, -1.5],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    return x, y


def compute_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    predicted_classes = torch.argmax(logits, dim=1)
    return (predicted_classes == target).float().mean().item()


def run_linear_forward_demo() -> dict[str, object]:
    """
    只做一次 forward，观察 nn.Linear 产生的 logits 如何接 CrossEntropyLoss。
    """
    torch.manual_seed(42)

    x, y = make_toy_classification_data()
    model = TinyLinearClassifier(in_features=2, num_classes=3)
    loss_fn = nn.CrossEntropyLoss()

    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
    loss = loss_fn(logits, y)
    predicted_classes = torch.argmax(logits, dim=1)

    return {
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "logits_shape": list(logits.shape),
        "y": y.tolist(),
        "logits": logits.detach().tolist(),
        "probabilities": probabilities.detach().tolist(),
        "predicted_classes": predicted_classes.detach().tolist(),
        "loss": loss.item(),
        "accuracy": compute_accuracy(logits, y),
    }


def run_one_training_step_demo() -> dict[str, object]:
    """
    只训练一步，用来观察：

    forward:  x -> model -> logits
    loss:     CrossEntropyLoss(logits, y)
    backward: loss.backward()
    update:   optimizer.step()

    重点看两件事：
    1. backward 后，linear.weight.grad 不再是 None
    2. step 后，loss 通常会下降一点
    """
    torch.manual_seed(42)

    x, y = make_toy_classification_data()
    model = TinyLinearClassifier(in_features=2, num_classes=3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    logits_before = model(x)
    loss_before = loss_fn(logits_before, y)

    optimizer.zero_grad()
    loss_before.backward()

    weight_grad = model.linear.weight.grad.detach().clone()
    bias_grad = model.linear.bias.grad.detach().clone()

    optimizer.step()

    logits_after = model(x)
    loss_after = loss_fn(logits_after, y)

    return {
        "loss_before_step": loss_before.item(),
        "loss_after_step": loss_after.item(),
        "weight_grad_shape": list(weight_grad.shape),
        "bias_grad_shape": list(bias_grad.shape),
        "weight_grad": weight_grad.tolist(),
        "bias_grad": bias_grad.tolist(),
    }


def run_minimal_training_loop_demo(
    epochs: int = 80,
    learning_rate: float = 0.1,
) -> dict[str, object]:
    """
    一个真正的最小分类训练闭环：

    数据 -> 模型 -> logits -> loss -> 反向传播 -> 参数更新 -> 评估
    """
    torch.manual_seed(42)

    x, y = make_toy_classification_data()
    model = TinyLinearClassifier(in_features=2, num_classes=3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    history: list[dict[str, object]] = []
    log_epochs = {1, 2, 3, 5, 10, 20, 40, epochs}

    for epoch in range(1, epochs + 1):
        # 1. forward：模型输出 logits
        logits = model(x)

        # 2. loss：CrossEntropyLoss 内部会做 log_softmax + NLLLoss
        loss = loss_fn(logits, y)

        # 3. backward：计算梯度
        optimizer.zero_grad()
        loss.backward()

        # 4. update：根据梯度更新参数
        optimizer.step()

        # 5. evaluate：训练后重新计算当前模型表现
        with torch.no_grad():
            eval_logits = model(x)
            eval_loss = loss_fn(eval_logits, y)
            eval_accuracy = compute_accuracy(eval_logits, y)
            predicted_classes = torch.argmax(eval_logits, dim=1)

        if epoch in log_epochs:
            history.append(
                {
                    "epoch": epoch,
                    "loss": eval_loss.item(),
                    "accuracy": eval_accuracy,
                    "predicted_classes": predicted_classes.tolist(),
                }
            )

    with torch.no_grad():
        final_logits = model(x)
        final_probabilities = torch.softmax(final_logits, dim=1)
        final_predicted_classes = torch.argmax(final_logits, dim=1)
        final_loss = loss_fn(final_logits, y)
        final_accuracy = compute_accuracy(final_logits, y)

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "history": history,
        "final_logits": final_logits.tolist(),
        "final_probabilities": final_probabilities.tolist(),
        "final_predicted_classes": final_predicted_classes.tolist(),
        "final_loss": final_loss.item(),
        "final_accuracy": final_accuracy,
        "learned_weight": model.linear.weight.detach().tolist(),
        "learned_bias": model.linear.bias.detach().tolist(),
    }


# ============================================================
# 5. 打印工具：让输出更适合学习观察
# ============================================================

def round_nested(value: object, digits: int = 4) -> object:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [round_nested(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: round_nested(item, digits) for key, item in value.items()}
    return value


def print_section(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


if __name__ == "__main__":
    print_section("1. hand-written logits -> CrossEntropyLoss")
    for result in run_softmax_cross_entropy_demo():
        print("case:", result["name"])
        print("logits:", round_nested(result["logits"]))
        print("probabilities:", round_nested(result["probabilities"]))
        print("target:", result["target"])
        print("predicted_class:", result["predicted_class"])
        print("target_probability:", round_nested(result["target_probability"]))
        print("loss:", round_nested(result["loss"]))
        print()

    print_section("2. batch shape demo")
    batch_result = run_batch_shape_demo()
    for key, value in batch_result.items():
        print(f"{key}:", round_nested(value))
    print()

    print_section("3. logits vs manually-softmaxed probabilities")
    loss_comparison = run_logits_vs_probabilities_demo()
    for key, value in loss_comparison.items():
        print(f"{key}:", round_nested(value))
    print()

    print_section("4. nn.Linear forward demo: x -> model -> logits -> loss")
    forward_result = run_linear_forward_demo()
    for key, value in forward_result.items():
        print(f"{key}:", round_nested(value))
    print()

    print_section("5. one training step demo: backward + optimizer.step")
    one_step_result = run_one_training_step_demo()
    for key, value in one_step_result.items():
        print(f"{key}:", round_nested(value))
    print()

    print_section("6. minimal training loop demo")
    training_result = run_minimal_training_loop_demo()
    print("training history:")
    for row in training_result["history"]:
        print(round_nested(row))
    print()
    print("final_predicted_classes:", training_result["final_predicted_classes"])
    print("target_classes:", training_result["y"])
    print("final_loss:", round_nested(training_result["final_loss"]))
    print("final_accuracy:", round_nested(training_result["final_accuracy"]))
    print("final_probabilities:", round_nested(training_result["final_probabilities"]))
    print("learned_weight shape: [num_classes, in_features]")
    print("learned_weight:", round_nested(training_result["learned_weight"]))
    print("learned_bias:", round_nested(training_result["learned_bias"]))
