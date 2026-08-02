"""Cat-face bounding-box detector: ResNet18 backbone + 4-output regression head.

Trained on CatFLW's own ground-truth bounding_boxes labels so it matches the
exact crop convention LandmarkNet expects (see dataset.py's box_padding). Runs
before LandmarkNet at inference to turn an arbitrary photo into a face crop,
instead of assuming the whole input image already is one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class BBoxNet(nn.Module):
    """Predicts a normalized [x_min, y_min, x_max, y_max] box for the input image."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, 4)
        self.backbone = backbone

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))
