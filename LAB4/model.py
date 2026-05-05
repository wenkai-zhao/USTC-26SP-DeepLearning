from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


EncoderName = Literal["resnet18", "mobilenet_v2"]


class ResNet18Encoder(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        try:
            backbone = models.resnet18(weights=None)
        except TypeError:
            backbone = models.resnet18(pretrained=False)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, feature_dim)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MobileNetV2Encoder(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        try:
            backbone = models.mobilenet_v2(weights=None)
        except TypeError:
            backbone = models.mobilenet_v2(pretrained=False)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, feature_dim))
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_encoder(name: EncoderName, feature_dim: int = 128) -> nn.Module:
    if name == "resnet18":
        return ResNet18Encoder(feature_dim)
    if name == "mobilenet_v2":
        return MobileNetV2Encoder(feature_dim)
    raise ValueError("encoder must be 'resnet18' or 'mobilenet_v2'.")


def build_projection_head(
    feature_dim: int = 128,
    projection_dim: int = 64,
    variant: str = "plain",
) -> nn.Sequential:
    if variant == "plain":
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )
    if variant == "batchnorm":
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )
    if variant == "wide":
        hidden_dim = feature_dim * 2
        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
    raise ValueError("projection variant must be one of: plain, batchnorm, wide.")


class SimCLRModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderName = "resnet18",
        feature_dim: int = 128,
        projection_dim: int = 64,
        projection_variant: str = "plain",
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(encoder, feature_dim)
        self.projection_head = build_projection_head(
            feature_dim, projection_dim, projection_variant
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections


class LinearClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int = 128,
        num_classes: int = 2,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)
