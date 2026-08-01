"""PyTorch dataset for the CatFLW facial-landmark dataset.

Expects the on-disk layout the CatFLW download extracts to:
    <root>/images/*.png
    <root>/labels/*.json   (one per image, same stem)

Each label JSON looks like:
    {"labels": [[x, y], ...],  # 48 points, pixel coordinates
     "bounding_boxes": [xmin, ymin, xmax, ymax]}
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

NUM_LANDMARKS = 48


class CatFLWDataset(Dataset):
    """Crops each image to its face bounding box and yields (image, flattened landmarks).

    Landmarks are returned as a (NUM_LANDMARKS * 2,) float32 array normalized to [0, 1]
    relative to the (padded) crop, so they're independent of the original image size.

    When augment=True, each sample additionally gets a random small rotation, a random
    jitter of the crop box (translation + scale), and color jitter -- all landmark-safe
    (rotation carries the points along with it). Horizontal flip is deliberately not
    offered: it requires knowing which of the 48 landmark indices are left/right pairs
    to swap, and that ordering isn't documented by CatFLW, so a naive flip would
    silently mislabel every point.
    """

    def __init__(
        self,
        root,
        image_size: int = 256,
        transform=None,
        box_padding: float = 0.15,
        augment: bool = False,
        max_rotation_deg: float = 15.0,
        box_jitter: float = 0.1,
    ):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.image_size = image_size
        self.transform = transform
        self.box_padding = box_padding
        self.augment = augment
        self.max_rotation_deg = max_rotation_deg
        self.box_jitter = box_jitter
        self.color_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        self.samples = sorted(p.stem for p in self.images_dir.glob("*.png"))
        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem = self.samples[idx]
        image = Image.open(self.images_dir / f"{stem}.png").convert("RGB")
        label = json.loads((self.labels_dir / f"{stem}.json").read_text())

        x_min, y_min, x_max, y_max = label["bounding_boxes"]
        box_w, box_h = x_max - x_min, y_max - y_min
        pad_x, pad_y = box_w * self.box_padding, box_h * self.box_padding
        shift_x = shift_y = 0.0

        if self.augment:
            pad_scale = 1.0 + random.uniform(-self.box_jitter, self.box_jitter)
            pad_x, pad_y = pad_x * pad_scale, pad_y * pad_scale
            shift_x = random.uniform(-self.box_jitter, self.box_jitter) * box_w
            shift_y = random.uniform(-self.box_jitter, self.box_jitter) * box_h

        x_min = max(0.0, x_min - pad_x + shift_x)
        y_min = max(0.0, y_min - pad_y + shift_y)
        x_max = min(float(image.width), x_max + pad_x + shift_x)
        y_max = min(float(image.height), y_max + pad_y + shift_y)

        crop = image.crop((x_min, y_min, x_max, y_max))
        crop_w, crop_h = crop.size

        # points in crop-local pixel coordinates
        points = np.array(label["labels"], dtype=np.float32) - [x_min, y_min]

        if self.augment and self.max_rotation_deg > 0:
            angle_deg = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            crop = crop.rotate(
                angle_deg, resample=Image.BILINEAR, center=(crop_w / 2, crop_h / 2)
            )
            theta = np.radians(angle_deg)
            cx, cy = crop_w / 2, crop_h / 2
            dx, dy = points[:, 0] - cx, points[:, 1] - cy
            points[:, 0] = cx + dx * np.cos(theta) + dy * np.sin(theta)
            points[:, 1] = cy - dx * np.sin(theta) + dy * np.cos(theta)

        if self.augment:
            crop = self.color_jitter(crop)

        crop = crop.resize((self.image_size, self.image_size), Image.BILINEAR)

        points[:, 0] /= crop_w
        points[:, 1] /= crop_h
        points = np.clip(points, 0.0, 1.0)

        if self.transform:
            crop = self.transform(crop)

        return crop, points.reshape(-1)
