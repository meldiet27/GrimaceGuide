"""Landmark heatmap model: a ResNet18 backbone with a deconvolution head.

Predicts one spatial heatmap per landmark instead of directly regressing (x, y)
coordinates. Direct-coordinate regression (plain MSE over a flat 96-unit output)
let the model minimize loss by predicting the mean position for CatFLW's
tightly-clustered points (eyes/nose/muzzle sit close together with low positional
variance across aligned crops) instead of learning to discriminate individual
points -- confirmed by visualizing predictions, where those points collapsed to a
blob while the higher-variance ear-tip points tracked fine. Heatmap regression
sidesteps this structurally: each landmark gets its own spatial output channel, so
there's no shared coordinate to collapse onto.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from ml.dataset import NUM_LANDMARKS


class LandmarkNet(nn.Module):
    """Predicts a (NUM_LANDMARKS, H, W) heatmap stack for an input crop.

    H = W = image_size // HEATMAP_STRIDE (see ml/dataset.py). Three stride-2
    deconv blocks undo 3 of the 5 stride-2 stages in the ResNet18 encoder,
    bringing stride-32 backbone features back down to stride-4.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.decoder = nn.Sequential(
            self._deconv_block(512, 256),
            self._deconv_block(256, 256),
            self._deconv_block(256, 256),
        )
        self.head = nn.Conv2d(256, NUM_LANDMARKS, kernel_size=1)

    @staticmethod
    def _deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(self.decoder(features))


def decode_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """Argmax-decodes a (B, NUM_LANDMARKS, H, W) heatmap stack to normalized (x, y).

    Returns a (B, NUM_LANDMARKS * 2) tensor in [0, 1], matching the flattened
    point layout CatFLWDataset uses for ground truth.
    """
    b, n, h, w = heatmaps.shape
    flat = heatmaps.view(b, n, h * w)
    idx = flat.argmax(dim=-1)
    ys = (idx // w).float() / (h - 1)
    xs = (idx % w).float() / (w - 1)
    return torch.stack([xs, ys], dim=-1).view(b, n * 2)
