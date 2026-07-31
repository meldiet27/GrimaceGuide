"""Landmark regression model: a ResNet18 backbone with a regression head."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from ml.dataset import NUM_LANDMARKS


class LandmarkNet(nn.Module):
    """Predicts normalized (x, y) for each of the NUM_LANDMARKS facial points."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, NUM_LANDMARKS * 2)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
