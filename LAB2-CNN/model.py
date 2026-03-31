from typing import List, Literal

import torch
import torch.nn as nn


PoolType = Literal["max", "avg"]


def build_pooling(name: PoolType) -> type[nn.Module]:
    if name == "max":
        return nn.MaxPool2d
    if name == "avg":
        return nn.AvgPool2d
    raise ValueError(f"Unsupported pooling type: {name}")


class CNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        conv_blocks: int = 2,
        base_channels: int = 32,
        kernel_size: int = 3,
        pool_type: PoolType = "max",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if conv_blocks < 1:
            raise ValueError("conv_blocks must be at least 1.")
        if kernel_size not in {3, 5, 7}:
            raise ValueError("kernel_size must be one of 3, 5, or 7.")

        padding = kernel_size // 2
        pool_cls = build_pooling(pool_type)

        layers: List[nn.Module] = []
        in_channels = 1
        out_channels = base_channels

        for _ in range(conv_blocks):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    pool_cls(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
            out_channels *= 2

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
