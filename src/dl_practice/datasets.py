from __future__ import annotations

import torch
from torch.utils.data import TensorDataset


def make_linear_regression_data(
    n: int = 200,
    weight: float = 2.0,
    bias: float = 1.0,
    noise_std: float = 0.2,
    seed: int = 42,
) -> TensorDataset:
    """Create a tiny synthetic y = wx + b regression dataset."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=generator)
    noise = noise_std * torch.randn(n, 1, generator=generator)
    y = weight * x + bias + noise
    return TensorDataset(x, y)
