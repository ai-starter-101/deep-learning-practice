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
