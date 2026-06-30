from __future__ import annotations

import torch
import torch.nn as nn


def run_softmax_cross_entropy_demo() -> list[dict[str, object]]:
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


if __name__ == "__main__":
    for result in run_softmax_cross_entropy_demo():
        print("case:", result["name"])
        print("logits:", result["logits"])
        print("probabilities:", result["probabilities"])
        print("target:", result["target"])
        print("predicted_class:", result["predicted_class"])
        print("target_probability:", result["target_probability"])
        print("loss:", result["loss"])
        print()

    batch_result = run_batch_shape_demo()
    print("case: batch_size_greater_than_one")
    print("logits_shape:", batch_result["logits_shape"])
    print("target_shape:", batch_result["target_shape"])
    print("logits:", batch_result["logits"])
    print("probabilities:", batch_result["probabilities"])
    print("target:", batch_result["target"])
    print("predicted_classes:", batch_result["predicted_classes"])
    print("per_sample_loss:", batch_result["per_sample_loss"])
    print("mean_loss:", batch_result["mean_loss"])
    print()

    loss_comparison = run_logits_vs_probabilities_demo()
    print("case: logits_vs_manual_softmax_input")
    print("correct_loss_from_logits:", loss_comparison["correct_loss_from_logits"])
    print("loss_after_manual_softmax:", loss_comparison["loss_after_manual_softmax"])
