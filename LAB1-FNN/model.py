from typing import List

import torch
import torch.nn as nn


def build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "swish":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class FNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        current = input_dim
        act = build_activation(activation)

        for h in hidden_dims:
            layers.append(nn.Linear(current, h))
            layers.append(act)
            current = h

        layers.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
