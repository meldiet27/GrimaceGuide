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


def decode_heatmaps(heatmaps: torch.Tensor, mode: str = "parabolic") -> torch.Tensor:
    """Decodes a (B, NUM_LANDMARKS, H, W) heatmap stack to normalized (x, y).

    Returns a (B, NUM_LANDMARKS * 2) tensor in [0, 1], matching the flattened
    point layout CatFLWDataset uses for ground truth.

    Plain argmax can only ever return a peak on the integer heatmap grid, so at
    HEATMAP_STRIDE=4 it quantizes every point to 1/64th of the crop no matter how
    good the model is. That floor is a large share of the error budget for the AUs
    that measure short distances (see docs/heatmap_decoding.md), so the peak is
    refined to sub-pixel precision by default:

      argmax     -- integer grid position (the original behavior)
      quarter    -- shifts a quarter-pixel toward the larger neighbor (SimpleBaseline's trick)
      parabolic  -- fits a parabola through the peak and its two neighbors per axis
      soft       -- intensity-weighted centroid over a 5x5 window around the peak

    parabolic is the default: targets are rendered Gaussians (ml/dataset.py), and a
    parabola is an exact fit to a Gaussian's neighborhood in the limit, so it recovers
    the true peak rather than approximating it with a fixed step. Refinement is applied
    post-hoc at decode time -- no retraining is needed to benefit from it.
    """
    b, n, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, n, h * w)
    idx = flat.argmax(dim=-1)
    px, py = idx % w, idx // w

    def value_at(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
        clamped = (iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)).unsqueeze(-1)
        return flat.gather(-1, clamped).squeeze(-1)

    zeros = torch.zeros_like(flat[..., 0])
    if mode == "argmax":
        dx = dy = zeros
    elif mode == "soft":
        radius = 2
        weight_sum, x_sum, y_sum = zeros.clone(), zeros.clone(), zeros.clone()
        for offset_y in range(-radius, radius + 1):
            for offset_x in range(-radius, radius + 1):
                ix, iy = px + offset_x, py + offset_y
                in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
                weight = torch.where(in_bounds, value_at(ix, iy).clamp(min=0), zeros)
                weight_sum = weight_sum + weight
                x_sum = x_sum + weight * offset_x
                y_sum = y_sum + weight * offset_y
        safe = weight_sum.clamp(min=1e-6)
        dx, dy = x_sum / safe, y_sum / safe
    else:
        center = value_at(px, py)
        left, right = value_at(px - 1, py), value_at(px + 1, py)
        up, down = value_at(px, py - 1), value_at(px, py + 1)
        if mode == "quarter":
            dx, dy = 0.25 * torch.sign(right - left), 0.25 * torch.sign(down - up)
        elif mode == "parabolic":
            # Peak of the parabola through (-1, left), (0, center), (1, right). The
            # denominator is negative at a true maximum; where the sampled neighbourhood
            # isn't concave (a flat or saddle region) the offset isn't meaningful, so
            # it falls back to the unrefined integer peak.
            eps = 1e-6
            den_x, den_y = left + right - 2 * center, up + down - 2 * center
            dx = torch.where(den_x < -eps, 0.5 * (left - right) / den_x.clamp(max=-eps), zeros)
            dy = torch.where(den_y < -eps, 0.5 * (up - down) / den_y.clamp(max=-eps), zeros)
            dx, dy = dx.clamp(-0.5, 0.5), dy.clamp(-0.5, 0.5)
        else:
            raise ValueError(f"Unknown decode mode: {mode!r}")

        # A refined offset needs both neighbours to exist; peaks on the border keep the
        # integer position rather than extrapolating off the edge of the heatmap.
        dx = torch.where((px > 0) & (px < w - 1), dx, zeros)
        dy = torch.where((py > 0) & (py < h - 1), dy, zeros)

    xs = (px.float() + dx) / (w - 1)
    ys = (py.float() + dy) / (h - 1)
    return torch.stack([xs, ys], dim=-1).view(b, n * 2)
