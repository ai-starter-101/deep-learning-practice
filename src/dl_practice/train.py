from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dl_practice.datasets import make_linear_regression_data
from dl_practice.models import LinearRegressionModel


def train_linear_regression(epochs: int = 80, lr: float = 0.05) -> dict[str, float]:
    dataset = make_linear_regression_data()
    train_set, val_set = random_split(
        dataset,
        [160, 40],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=40)

    model = LinearRegressionModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            val_loss = evaluate_loss(model, val_loader, loss_fn)
            print(f"epoch={epoch + 1:03d} val_loss={val_loss:.4f}")

    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    val_loss = evaluate_loss(model, val_loader, loss_fn)
    return {"weight": weight, "bias": bias, "val_loss": val_loss}


def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * len(x)
            total_count += len(x)
    return total_loss / total_count


if __name__ == "__main__":
    metrics = train_linear_regression()
    print(metrics)
